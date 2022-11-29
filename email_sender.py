from pathlib import Path
from smtplib import SMTPException, SMTP_SSL
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailSender:
    """Stores email sender, reciever, and pw."""

    def __init__(
        self,
        sender: str = None,
        reciever: str = None,
        pw: str = None,
        send: bool = True,
        **kwargs
    ) -> None:
        """Store email params."""

        self.sender = sender
        self.reciever = reciever
        self.pw = self.retrieve_pw(pw)
        self.send = send
        if self.send is not None:
            print(f"Email settings: send set to {send}")
        # dummy function in case an argument is not provided:
        if None in (sender, reciever, pw):
            print(
                "At least one of email sender, reciever, or pw was not"
                "specified, will not send any emails."
            )
            self.email = lambda subject, content: 0
        else:
            self.email = self._email

    def retrieve_pw(self, file: str=None) -> str:
        """Retrieves the gmail password from a file."""

        if file is None:
            return None
        with open(Path()/file, 'r') as pw_file:
            pw = pw_file.read()
        return pw

    def _email(self, subject, content=""):
        """Send an email."""
        email(
            content=content,
            subject=subject,
            sender=self.sender,
            reciever=self.reciever,
            pw=self.pw,
            send=self.send,
        )


def email(
    content: str,
    subject: str,
    sender: str,
    reciever: str,
    pw: str = None,
    send: bool = True,
) -> None:
    """
    Sends an email from a gmail account.
    :param content: the message inside the email.
    :param subject: the subject line.
    :param sender: the sending email address.
    :param reciever: the destination email address.
    :param pw: the gmail password for the sending email address.
    :param send: will only send an email if this is true.
    :return: None
    """

    if not send:
        return

    message = MIMEMultipart()
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = reciever
    message.attach(MIMEText(content, "plain"))

    try:
        context = ssl.create_default_context()
        with SMTP_SSL(host="smtp.gmail.com", port=465, context=context) as server:
            server.login(sender, pw)
            server.sendmail(sender, reciever, message.as_string())
            server.quit()
    except Exception as e:
        print("Error while trying to send email: \n%s", e)


