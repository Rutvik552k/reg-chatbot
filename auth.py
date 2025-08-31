
import uuid
import hashlib
import secrets
from typing import Optional, Tuple, Dict

import streamlit as st
from db import init_db, create_user_row, read_user_by_email

def hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    if not salt:
        salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000, dklen=32)
    return salt, dk.hex()

def verify_password(password: str, salt: str, hashed_hex: str) -> bool:
    _, h = hash_password(password, salt)
    return secrets.compare_digest(h, hashed_hex)

def create_user(email: str, password: str):
    email = email.strip().lower()
    if not email or not password:
        return False, "Email and password required."
    salt, hashed = hash_password(password)
    user_id = str(uuid.uuid4())
    ok = create_user_row(user_id, email, salt, hashed)
    if ok:
        return True, user_id
    return False, "Email already registered."

def authenticate(email: str, password: str):
    email = email.strip().lower()
    row = read_user_by_email(email)
    if not row:
        return False, None, "No account with that email."
    uid, em, salt, ph = row
    if verify_password(password, salt, ph):
        return True, {"id": uid, "email": em}, ""
    return False, None, "Incorrect password."

def auth_gate() -> None:
    st.subheader("Login to continue")
    tab_login, tab_signup = st.tabs(["Login", "Sign up"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submit = st.form_submit_button("Login")
        if submit:
            ok, user, msg = authenticate(email, password)
            if ok:
                st.session_state.user = user
                st.success("Logged in!")
                st.rerun()
            else:
                st.error(msg)

    with tab_signup:
        with st.form("signup_form"):
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            submit = st.form_submit_button("Create account")
        if submit:
            ok, user_id_or_msg = create_user(email, password)
            if ok:
                st.success("Account created. Please log in.")
            else:
                st.error(user_id_or_msg)
