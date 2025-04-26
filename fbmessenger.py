
from rds import get_db_connection
import os
import requests
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
)

# Assuming your Facebook Page Access Token is stored as an environment variable
FB_PAGE_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
FB_GRAPH_API_URL = "https://graph.facebook.com/v22.0/me/messages"
FB_GRAPH_API_URL_FOR_ACCESS_TOKEN ="https://graph.facebook.com/v22.0"

def user_bot_id_page_id_mapping (page_id):
    con = get_db_connection()
    if con is None:
        print("Failed to establish a database connection.")
        return None, None

    try:
        cursor = con.cursor()
        sql_query = """
            SELECT bot_id, user_id FROM fb_messenger              
            WHERE page_id = %s
        """
        cursor.execute(sql_query, (page_id,))

        row = cursor.fetchone()

        if row:
            return row[0], row[1]
        else:
            return None, None

    except Exception as e:
        print(f"Database query error: {str(e)}")
        return None, None

    finally:
        cursor.close()
        con.close()



def send_fb_message(page_id, psid, bot_reply):
    try:
        # Fetch the Page Access Token first
        uri = f"{FB_GRAPH_API_URL_FOR_ACCESS_TOKEN}/{page_id}"
        params_for_token = {
            "fields": "access_token",
            "access_token": FB_PAGE_ACCESS_TOKEN  # System token to fetch Page token
        }
        
        res = requests.get(uri, params=params_for_token)
        res.raise_for_status()
        data = res.json()
        access_token = data.get("access_token")

        if not access_token:
            logging.error(f"No access token found for Page ID {page_id}. Response: {data}")
            return

    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to fetch access token for Page ID {page_id}: {e}")
        return

    headers = {
        "Content-Type": "application/json"
    }

    payload = {
        "recipient": {
            "id": psid
        },
        "message": {
            "text": bot_reply
        },
        "messaging_type": "RESPONSE"
    }

    params = {
        "access_token": access_token
    }

    try:
        response = requests.post(
            FB_GRAPH_API_URL,
            params=params,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        logging.info(f"Message sent to PSID {psid}. Response: {response.json()}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send message to PSID {psid}: {e}")
