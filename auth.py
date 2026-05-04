"""
Authentication Module
Handles registration, login, and password hashing.
Uses Python's built-in hashlib (no extra dependencies).

HR_SECRET_CODE must be entered during registration to get HR role.
"""

import hashlib
import secrets
from database import create_user, get_user_by_employee_id

HR_SECRET_CODE = "HR@2024"   # Change this to your preferred HR access code


def hash_password(password: str) -> str:
    """Hash a password with a random salt using PBKDF2-HMAC-SHA256."""
    salt   = secrets.token_hex(16)
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
    return f"{salt}:{hashed.hex()}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash."""
    try:
        salt, hashed = stored_hash.split(":", 1)
        new_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000)
        return new_hash.hex() == hashed
    except Exception:
        return False


def register(employee_id: str, name: str, password: str,
             department: str = "General", hr_code: str = "") -> dict:
    """
    Register a new user.
    Returns: { "success": bool, "message": str, "role": str }
    """
    employee_id = employee_id.strip().upper()
    name        = name.strip()
    password    = password.strip()

    if not employee_id or not name or not password:
        return {"success": False, "message": "All fields are required."}

    if len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters."}

    # Check if employee ID already taken
    if get_user_by_employee_id(employee_id):
        return {"success": False, "message": f"Employee ID '{employee_id}' is already registered."}

    if hr_code.strip() == "":
        role = "employee"
    elif hr_code.strip() == HR_SECRET_CODE:
        role = "hr"
    else:
        return {"success": False, "message": "Incorrect HR access code. If you are an employee, leave that field empty."}

    pw_hash = hash_password(password)
    ok      = create_user(employee_id, name, pw_hash, department, role)

    if ok:
        return {
            "success": True,
            "message": f"Account created! Role: {'HR' if role == 'hr' else 'Employee'}",
            "role": role,
        }
    return {"success": False, "message": "Registration failed. Please try again."}


def login(employee_id: str, password: str) -> dict:
    """
    Authenticate a user.
    Returns: { "success": bool, "user": dict | None, "message": str }
    """
    employee_id = employee_id.strip().upper()
    user        = get_user_by_employee_id(employee_id)

    if not user:
        return {"success": False, "user": None,
                "message": "Employee ID not found."}

    if not verify_password(password, user["password_hash"]):
        return {"success": False, "user": None,
                "message": "Incorrect password."}

    # Return user without password hash
    safe_user = {k: v for k, v in user.items() if k != "password_hash"}
    return {"success": True, "user": safe_user, "message": "Login successful."}