# pyre-strict
import uuid
import logging
from flask import Flask, Response, after_this_request, request, jsonify
from flask_cors import CORS
from openai import OpenAI
from pinecone import Pinecone
from flask_restful import Resource
from rds import get_db_connection
import cohere
import time
import base64
import json
import os
from whatsapp import user_bot_id_phone_mapping, send_whatsapp_message
from fbmessenger import user_bot_id_page_id_mapping, send_fb_message
from instamessenger import user_bot_instagram_id_mapping

from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, expose_headers=["uid", "doc_keys"])

OpenAI_KEY = os.getenv("OpenAI_KEY")
Pinecone_api_key = os.getenv("Pinecone_api_key")
vector_count = 8000
batch_size = 100
openai_client = OpenAI(api_key = OpenAI_KEY)
pinecone_client = Pinecone(
    api_key=Pinecone_api_key, environment="us-east-1", pool_threads=30
)
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(COHERE_API_KEY)

class IndexHandler(Resource):
    def __init__(self, index_name):
        self.index_name = index_name
        self.pc = Pinecone(
            api_key=Pinecone_api_key, environment="eu-west1-gcp", pool_threads=30
        )
        self.embed_model = "text-embedding-ada-002"
        self.limit = 5000
        self.index_pc = self.pc.Index(index_name)
        self.docs = []
        self.dimension = 1536
        self.S3BucketName = None

   
    def rerank_contexts(self, query, contexts, doc_keyss, file_type, sub_type):
        """
        Reranks the contexts using Cohere's rerank model.
        """
        try:
            response = cohere_client.rerank(
                model="rerank-english-v3.0",
                query=query,
                documents=contexts,
                top_n=10,
            )           
            reranked_documents = [contexts[result.index] for result in response.results]
            doc_keys = [doc_keyss[result.index] for result in response.results]
            file_types = [file_type[result.index] for result in response.results]
            sub_types = [sub_type[result.index] for result in response.results]
            logging.info(f'''doc_keys , {doc_keys}''')

            return reranked_documents, doc_keys[:4] or [], file_types[:4] or [], sub_types[:4] or []
        except Exception as e:
            logging.error(f"Error reranking contexts: {e}")
            return contexts, []

    def augment_query_generated(self, query, model="gpt-3.5-turbo"):
        """
        Augments the original query by generating a hypothetical answer using OpenAI's API.

        Parameters:
            query (str): The original user query.
            model (str): The model to be used for generating the answer.

        Returns:
            str: The generated hypothetical answer.
        """
        messages = [
            {
                "role": "system",
                "content": "You are a expert all purpose assistant. Provide an example answer to the given question, that might be found in a websites or official documents",
            },
            {"role": "user", "content": query},
        ]

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
        )
        content = response.choices[0].message.content
        return content or query  # ckeck back for validity

    def augment_query(self, query):
        """
        Augments the original query with a generated hypothetical answer.

        Parameters:
            query (str): The original user query.

        Returns:
            str: The combined original query and generated hypothetical answer.
        """
        hypothetical_answer = self.augment_query_generated(query)
        joint_query = f"{query} {hypothetical_answer}"
        return joint_query

    def system_promt (self,bot_id):
        con = get_db_connection()
        cursor = con.cursor()

        try:  
            
            # Get user credits and version, using SELECT ... FOR UPDATE to lock the row
            sql_query = """select prompt from customprompt where bot_id = %s"""
            cursor.execute(sql_query, (bot_id,))
            result = cursor.fetchone()

            print(result[0])
            con.commit()
            return result[0] or """You are a helpful and friendly assistant whose job is to faithfully
        answer the question based on the context below. You are serving as a helper to visitors to a
        website who could be potential customers for the business."""

        except Exception as e:
            con.rollback()  # Rollback if an error occurs
            print(f"Error occurred: {e}")
            return """You are a helpful and friendly assistant whose job is to faithfully
        answer the question based on the context below. You are serving as a helper to visitors to a
        website who could be potential customers for the business."""

        finally:
            cursor.close()
            con.close()


    def retrieve(self, query, S3BucketName, rerank, queryexpanstion,premsg):
        sys_prompt  =  self.system_promt(S3BucketName)
        parsedValue = 10
        if queryexpanstion:
            augmented_query = self.augment_query(query)
            query = str(augmented_query)
        res = openai_client.embeddings.create(input=str(query), model=self.embed_model)
        xq = res.data[0].embedding
        ress = self.index_pc.query(
            vector=xq, top_k=parsedValue, include_metadata=True, namespace=S3BucketName
        )
        contexts = [x["metadata"]["passagetext"] for x in ress["matches"]]               
        doc_keyss = [x["metadata"]["doc_key"] for x in ress["matches"]]
        file_type = [x["metadata"]["type"] for x in ress["matches"]]
        sub_type = [x["metadata"]["subtype"] for x in ress["matches"]]
        prompt_start = f"""{sys_prompt} \n\nContext:\n {premsg} \n"""

        print("prompt_start" ,prompt_start)
        prompt_end = f"\n\nQuestion: {query}\n Answer:"
        prompt = ""
       
        if rerank and contexts:
            reranked_contexts, doc_keys, file_types, sub_types = self.rerank_contexts(
                query, contexts, doc_keyss, file_type, sub_type
            )
            combined_contexts = []
            for context in reranked_contexts:
                temp_combined = combined_contexts + [context]
                if len("\n\n---\n\n".join(temp_combined)) >= self.limit:
                    break
                combined_contexts.append(context)
            prompt = prompt_start + "\n\n---\n\n".join(combined_contexts) + prompt_end
            return prompt, doc_keys, contexts, file_types, sub_types     
        combined_contexts = []
        for index, value in enumerate(contexts):
            temp_combined = combined_contexts + [
                value
            ]  # Include the current context in the combined contexts
            if (
                len("\n\n---\n\n".join(temp_combined)) >= self.limit
            ):  # Check if adding this context would exceed the limit
                break
            combined_contexts.append(value)

        prompt = prompt_start + "\n\n---\n\n".join(combined_contexts) + prompt_end
        return prompt, list(set(doc_keyss[:5])), contexts, file_type[:5], sub_type[:5]

index_handler = IndexHandler(index_name="jaano2")

def store_data_rag_analysis(
    uid, contexts, botid, answer, query, prompt, email, phone, name, visitor_id
):
    print("uid, contexts, botid, answer, query, prompt, email, phone, name",uid, contexts, botid, answer, query, prompt, email, phone, name)
    con = get_db_connection()
    if not visitor_id:
        visitor_id = str(uuid.uuid4()) 
        
    if con is None:
        logging.info("Failed to establish a database connection.")
        return
    cursor = con.cursor()
    try:
        # Insert into ragQuality
        sql_query = """
            INSERT INTO ragQuality (rag_quality_id, bot_id, user_query, prompt_generated, answer,email,phone,name, visitor_id)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        data = (uid, botid, query, prompt, answer, email, phone, name, visitor_id)
        cursor.execute(sql_query, data)
        # Insert into topkChunks
        sql_queryy = """
            INSERT INTO topkChunks (rag_quality_id, bot_id, chunk)
            VALUES (%s, %s, %s)
        """
        for context in contexts:
            data = (uid, botid, context)
            cursor.execute(sql_queryy, data)
        # Commit the transaction
        con.commit()
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        con.rollback()  # Rollback in case of error
    finally:
        cursor.close()
        con.close()
        print("successfully inserted into data-base")

def get_credits(user_id: str, bot_id: str, max_retries=100, retry_delay=0.5):
    con = get_db_connection()
    cursor = con.cursor()
    
    try:
        # Start a transaction
        con.begin()

        # Get user credits and version, using SELECT ... FOR UPDATE to lock the row
        sql_query = """SELECT credits, version FROM user_credits WHERE user_id = %s FOR UPDATE"""
        cursor.execute(sql_query, (user_id,))
        result = cursor.fetchone()
        if not result:
            raise ValueError(f"No user found with user_id: {user_id}")

        credits, version = result

        # Check if credits are available
        if credits <= 0:
            return False

        # Attempt to update credits
        sql_update_query = """UPDATE user_credits 
                              SET credits = %s, version = %s 
                              WHERE user_id = %s AND version = %s"""
        new_credits = credits - 1
        new_version = version + 1
        rows_updated = cursor.execute(sql_update_query, (new_credits, new_version, user_id, version))

        # Retry logic for version conflict (Optimistic Locking)
        retries = 0
        while rows_updated == 0 and retries < max_retries:
            # Wait before retrying
            time.sleep(retry_delay)

            # Re-fetch the latest data (credits and version)
            cursor.execute(sql_query, (user_id,))
            result = cursor.fetchone()
            if not result:
                raise ValueError(f"No user found with user_id: {user_id}")
            credits, version = result

            # Recalculate new values using the current data
            new_credits = credits - 1
            new_version = version + 1

            # Attempt to update again using the latest version
            rows_updated = cursor.execute(sql_update_query, (new_credits, new_version, user_id, version))
            retries += 1

        if rows_updated == 0:
            raise ValueError(f"Failed to update credits after {max_retries} retries due to concurrent modifications.")

        # Insert into CreditsHistory (same as before)
        sql_insert_query = """INSERT INTO CreditsHistory 
                              (user_id, bot_id, modifier, remaining_credits, reason) 
                              VALUES (%s, %s, %s, %s, %s)"""
        cursor.execute(sql_insert_query, (user_id, bot_id, "-1", new_credits, "API Call"))

        # Commit the transaction
        con.commit()
        return True

    except Exception as e:
        con.rollback()  # Rollback if an error occurs
        print(f"Error occurred: {e}")
        return False

    finally:
        cursor.close()
        con.close()

@app.route("/botquery/<user_id>/<botId>", methods=["POST"])
def bot_query(user_id,botId):
    reranking = (
        request.args.get("reranking", default=False, type=lambda x: x.lower() == "true")
        or False
    )
    queryexpanstion = (
        request.args.get(
            "queryexpanstion", default=False, type=lambda x: x.lower() == "true"
        )
        or False
    )

    result = get_credits(user_id,botId)

    if result == False:  # Check for zero or negative credits
        response_stream = "Your credits are 0. Please contact support for assistance."
    
        def generate_response():
            # Yield the response message in chunks
            yield response_stream

        # Stream the response
        response = Response(generate_response(), content_type="text/plain")
        return response , 200


    email = ""
    phone = ""
    name = ""

    data = request.get_json()
    que = data.get("que", "")
    visitor_id = data.get("visitor_id", "") 
    email = data.get("email", " ")
    phone = data.get("phone", " ")
    name = data.get("name", " ")
    premsg = data.get("premsg", "")
    # Generate a unique identifier
   
    uid = str(uuid.uuid4())
    prompt, doc_keys, contexts, file_type, sub_type  = index_handler.retrieve(que, botId,reranking,queryexpanstion,premsg)
    list_of_url_object = [{"url": doc_keys[i], "file_type": file_type[i], "sub_type": sub_type[i]} for i in range(len(doc_keys))]
    json_string = json.dumps(list_of_url_object)

    encoded_string = base64.b64encode(json_string.encode('utf-8')).decode('utf-8')  
    # response_stream, headers = index_handler.complete(prompt)

    def generate():
            stream = openai_client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=400,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None,
                stream=True,
            )
            # Iterate through the stream and yield the content
            for chunk in stream:                
                content = chunk.choices[0].delta.content                
                if content is not None:                  
                    yield content

    @after_this_request
    def store_data(response):
        # Store data for analysis
        store_data_rag_analysis(
            uid, contexts, botId, "", que, prompt, email, phone, name , visitor_id
        )
        return response
    
    return generate(), {"Content-Type": "text/plain", "doc_keys": encoded_string, "uid": uid} 

@app.route("/answer/<ansId>", methods=["POST"])
def bot_ans(ansId):
    data = request.get_json()
    ans = data["ans"]
    if not ans:
        logging.info("No answer provided in the request.")
        return "No answer provided", 400

    con = get_db_connection()
    if con is None:
        logging.info("Failed to establish a database connection.")
        return "Database connection failed", 500

    cursor = con.cursor()
    try:
        # Update the 'answer' field in ragQuality for the provided ansId
        sql_query = """
            UPDATE ragQuality 
            SET answer = %s 
            WHERE rag_quality_id = %s
        """
        data = (ans, ansId)
        cursor.execute(sql_query, data)
        con.commit()  # Commit the transaction

        logging.info(f"Answer updated for rag_quality_id {ansId}")
        return "Answer updated successfully", 200
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        con.rollback()  # Rollback in case of error
        return "Failed to update answer", 500
    finally:
        cursor.close()
        con.close()

@app.route("/context/<ansId>", methods=["GET"])
def bot_context(ansId):
    con = get_db_connection()
    if con is None:
        logging.info("Failed to establish a database connection.")
        return "Database connection failed", 500

    cursor = con.cursor()
    try:
        # Fetch 'chunk' field from topkChunks for the given ansId
        sql_query = """
            SELECT chunk FROM topkChunks              
            WHERE rag_quality_id = %s
        """
        data = (ansId,)  # Ensure it's a tuple
        cursor.execute(sql_query, data)  
        rows = cursor.fetchall()       
        con.commit()  

        logging.info(f"Data retrieved for rag_quality_id {ansId}")
        return jsonify({"chunks": [i[0] for i in rows]}), 200
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        con.rollback()  
        return "Failed to retrieve data", 500
    finally:
        cursor.close()
        con.close()

@app.route('/webhook', methods=['GET'])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == "vedantkiranfarde":
        return challenge, 200  # Return the challenge value
    else:
        return jsonify({"error": "Verification failed"}), 403


@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    data = request.get_json()
    # Generate UID at the start
    uid = str(uuid.uuid4())
    logging.info(f'''data \n {data} \n''')    

    # Check if the payload contains the correct structure
    if not data or "entry" not in data or not isinstance(data["entry"], list):
        logging.error("Invalid payload structure: Missing 'entry'")
        return jsonify({"error": "Invalid payload structure"}), 400
    
    for entry in data["entry"]:
        print(entry)
        if "changes" not in entry or not isinstance(entry["changes"], list):
                logging.error("Invalid payload structure: Missing 'changes'")
                continue 
        for change in entry["changes"]:
            print(change)
            value = change.get("value", {})
            print("\n",value)
            print("\n",isinstance(value, dict))

            if not isinstance(value, dict):
                logging.error("Invalid 'value' structure")
                continue

            print("\n",value.get("metadata"))
            print("\n",value.get("messages"))

            if "messages" in value and "metadata" in value:
                metadata = value.get("metadata", {})
                phone_number_id = metadata.get("phone_number_id", "Unknown")
                print("\n",metadata)
                print("\n",phone_number_id)


                for message in value.get("messages", []):
                    sender_number = message.get("from", "")
                    user_message = message.get("text", {}).get("body", "")
                    visitor_id = sender_number  

                    logging.info(f"Message from {sender_number}: {user_message}")

                    # Identify bot namespace
                    bot_id, user_id = user_bot_id_phone_mapping(phone_number_id)

                    # Fetch relevant context from Pinecone
                    retrieval_result = index_handler.retrieve(user_message, bot_id, True, False, "")

                    # Ensure retrieval_result contains expected values
                    if retrieval_result and len(retrieval_result) >= 5:
                        prompt, doc_keys, contexts, file_type, sub_type = retrieval_result
                    else:
                        contexts, prompt, doc_keys, file_type, sub_type = [], "", [], [], []  # Default values

                    list_of_url_object = [
                        {"url": doc_keys[i], "file_type": file_type[i], "sub_type": sub_type[i]}
                        for i in range(len(doc_keys))
                    ]
                    json_string = json.dumps(list_of_url_object)
                    encoded_string = base64.b64encode(json_string.encode("utf-8")).decode("utf-8")

                    # Generate AI response
                    bot_response = openai_client.chat.completions.create(
                        model="gpt-4o-2024-08-06",
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0,
                        max_tokens=400,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                        stream=False,
                    )

                    bot_reply = bot_response.choices[0].message.content
                    print(bot_reply)

                    # Send response back to WhatsApp
                    send_whatsapp_message(phone_number_id, sender_number, bot_reply)

                    # Ensure `contexts` is captured correctly in `store_data`
                    # uid, contexts, botid, answer, query, prompt, email, phone, name, visitor_id

                    @after_this_request
                    def store_data(response, contexts=contexts, bot_reply=bot_reply, prompt=prompt, visitor_id=visitor_id):
                        store_data_rag_analysis(
                            uid, contexts, bot_id, bot_reply, user_message, prompt, "", visitor_id, "", visitor_id
                        )
                        return response

                return jsonify({"status": "received"}), 200

    return jsonify({"status": "invalid request"}), 400  # Handle incorrect payloads properly


@app.route('/fbmsg/webhook', methods=['GET'])
def verify_fb_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == "vedantkiranfarde":
        return challenge, 200  # Return the challenge value
    else:
        return jsonify({"error": "Verification failed"}), 403


@app.route("/fbmsg/webhook", methods=["POST"])
def fbmsg_webhook():
    data = request.get_json()
    uid = str(uuid.uuid4())
    logging.info(f"Webhook Event UID: {uid}")
    logging.info(f"Received Data: {json.dumps(data, indent=2)}")

    # Check if event is from a page subscription
    if data.get("object") == "page":
        for entry in data.get("entry", []):
            for event in entry.get("messaging", []):
                sender = event.get("sender", {})
                recipient = event.get("recipient", {})
                psid = sender.get("id")
                page_id = recipient.get("id")
                user_message = event.get("message", {}).get("text", "")
                message_is_echo = event.get("message", {}).get("is_echo", False)

                # Skip echo, delivery, or read events
                if message_is_echo or "delivery" in event or "read" in event:
                    logging.info("Skipped echo/delivery/read event.")
                    continue

                if not psid or not user_message:
                    logging.warning("Missing PSID or message.")
                    continue

                logging.info(f"Message from PSID {psid}: {user_message}")

                # Identify bot namespace (assume similar logic for bot_id lookup)
                bot_id, user_id = user_bot_id_page_id_mapping(page_id)

                # Retrieve context from Pinecone
                retrieval_result = index_handler.retrieve(user_message, bot_id, True, False, "")

                if retrieval_result and len(retrieval_result) >= 5:
                    prompt, doc_keys, contexts, file_type, sub_type = retrieval_result
                else:
                    contexts, prompt, doc_keys, file_type, sub_type = [], "", [], [], []

                list_of_url_object = [
                    {"url": doc_keys[i], "file_type": file_type[i], "sub_type": sub_type[i]}
                    for i in range(len(doc_keys))
                ]
                encoded_string = base64.b64encode(json.dumps(list_of_url_object).encode("utf-8")).decode("utf-8")

                # Generate AI response
                bot_response = openai_client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=400,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                bot_reply = bot_response.choices[0].message.content
                logging.info(f"Bot reply: {bot_reply}")

                # Send response back to FB Messenger
                send_fb_message(page_id, psid, bot_reply)

                # Store data after the response
                @after_this_request
                def store_data(response, contexts=contexts, bot_reply=bot_reply, prompt=prompt, visitor_id=psid):
                    store_data_rag_analysis(
                        uid, contexts, bot_id, bot_reply, user_message, prompt, "", visitor_id, "", visitor_id
                    )
                    return response

        return jsonify({"status": "received"}), 200

    return jsonify({"status": "invalid request"}), 400


@app.route('/insta/webhook', methods=['GET'])
def verify_insta_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")

    if mode == "subscribe" and token == "vedantkiranfarde":
        return challenge, 200  # Return the challenge value
    else:
        return jsonify({"error": "Verification failed"}), 403

@app.route("/insta/webhook", methods=["POST"])
def insta_webhook():
    data = request.get_json()
    uid = str(uuid.uuid4())
    logging.info(f"Webhook Event UID: {uid}")
    logging.info(f"Received Data: {json.dumps(data, indent=2)}")

    # Check if event is from a page subscription
    if data.get("object") == "page":
        for entry in data.get("entry", []):
            for event in entry.get("messaging", []):
                sender = event.get("sender", {})
                recipient = event.get("recipient", {})
                psid = sender.get("id")
                instagram_id = recipient.get("id")
                user_message = event.get("message", {}).get("text", "")
                message_is_echo = event.get("message", {}).get("is_echo", False)

                # Skip echo, delivery, or read events
                if message_is_echo or "delivery" in event or "read" in event:
                    logging.info("Skipped echo/delivery/read event.")
                    continue

                if not psid or not user_message:
                    logging.warning("Missing PSID or message.")
                    continue

                logging.info(f"Message from PSID {psid}: {user_message}")

                # Identify bot namespace (assume similar logic for bot_id lookup)
                bot_id, user_id, page_id = user_bot_instagram_id_mapping(instagram_id)

                # Retrieve context from Pinecone
                retrieval_result = index_handler.retrieve(user_message, bot_id, True, False, "")

                if retrieval_result and len(retrieval_result) >= 5:
                    prompt, doc_keys, contexts, file_type, sub_type = retrieval_result
                else:
                    contexts, prompt, doc_keys, file_type, sub_type = [], "", [], [], []

                list_of_url_object = [
                    {"url": doc_keys[i], "file_type": file_type[i], "sub_type": sub_type[i]}
                    for i in range(len(doc_keys))
                ]
                encoded_string = base64.b64encode(json.dumps(list_of_url_object).encode("utf-8")).decode("utf-8")

                # Generate AI response
                bot_response = openai_client.chat.completions.create(
                    model="gpt-4o-2024-08-06",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=400,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )

                bot_reply = bot_response.choices[0].message.content
                logging.info(f"Bot reply: {bot_reply}")

                # Send response back to FB Messenger
                send_fb_message(page_id, instagram_id, psid, bot_reply)

                # Store data after the response
                @after_this_request
                def store_data(response, contexts=contexts, bot_reply=bot_reply, prompt=prompt, visitor_id=psid):
                    store_data_rag_analysis(
                        uid, contexts, bot_id, bot_reply, user_message, prompt, "", visitor_id, "", visitor_id
                    )
                    return response

        return jsonify({"status": "received"}), 200

    return jsonify({"status": "invalid request"}), 400
                
    


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=True)  # ,


