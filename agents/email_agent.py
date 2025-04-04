import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

SENDGRID_API_KEY = os.getenv("SENDGRID_API_KEY")

def send_email(to_email: str, subject: str, body: str):
    message = Mail(
        from_email="mahithchigurupati@gmail.com",
        to_emails=to_email,
        subject=subject,
        plain_text_content=body,
    )
    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        return {
            "status": "success" if response.status_code == 202 else "failure",
            "response": "Email sent successfully." if response.status_code == 202 else "Failed to send email.",
        }
    except Exception as e:
        return {"status": "error", "response": str(e)}
