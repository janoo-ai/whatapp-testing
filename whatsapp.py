from rds import get_db_connection
import os
import requests

def user_bot_id_phone_mapping(phone_number_id):
    con = get_db_connection()
    if con is None:
        print("Failed to establish a database connection.")
        return None, None

    try:
        cursor = con.cursor()
        sql_query = """
            SELECT bot_id, user_id FROM whatsapp_integration              
            WHERE phone_number_id = %s
        """
        cursor.execute(sql_query, (phone_number_id,))

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



def send_whatsapp_message(phone_number_id, sender_number, bot_response:str):
    WHATSAPP_API_URL = f"https://graph.facebook.com/v22.0/{phone_number_id}/messages"
    WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN") 

    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }
    
    data = {
        "messaging_product": "whatsapp",
        "to": sender_number,
        "text": {"body": bot_response}
    }

    try:
        response = requests.post(WHATSAPP_API_URL, json=data, headers=headers)
        response_data = response.json()

        if response.status_code == 200:
            print(f"✅ Message sent successfully to {sender_number}")
        else:
            print(f"❌ Failed to send message. Response: {response_data}")
        
        return response_data  # Returning response for debugging purposes

    except requests.exceptions.RequestException as e:
        print(f"❌ Error while sending message: {e}")
        return None
    
    

