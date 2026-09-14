from pathlib import Path
import pandas as pd


# ============================================================
# LOAD EMPLOYEE DATA
# ============================================================

EMPLOYEE_FILE = Path("data/employees.csv")

employees = pd.read_csv(EMPLOYEE_FILE)


# ============================================================
# TOOL 1: CHECK EMPLOYEE ELIGIBILITY
# ============================================================

def check_employee_eligibility(employee_id):
    """
    Check whether an employee is eligible for business travel.
    """

    employee_id = employee_id.strip().upper()

    employee = employees[
        employees["employee_id"] == employee_id
    ]

    if employee.empty:
        return {
            "status": "Not Found",
            "employee_id": employee_id,
            "message": "Employee ID does not exist in the employee database."
        }

    employee = employee.iloc[0]

    return {
        "status": employee["eligibility_status"],
        "employee_id": employee["employee_id"],
        "country": employee["country"],
        "employee_type": employee["employee_type"],
        "department": employee["department"]
    }


# ============================================================
# TEST TOOL
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("EMPLOYEE ELIGIBILITY TOOL")
    print("=" * 60)

    test_ids = [
        "EMP001",
        "EMP003",
        "EMP999"
    ]

    for employee_id in test_ids:

        result = check_employee_eligibility(employee_id)

        print("\nEmployee:", employee_id)
        print("Result:", result)
# ============================================================
# TOOL 2: TRIP VALIDATION
# ============================================================

def validate_trip(employee_id, trip_type, amount, time):
    """
    Validate a business trip against the employee's
    country-specific travel policy limit.
    """

    # Check employee eligibility first
    employee_result = check_employee_eligibility(employee_id)

    if employee_result["status"] == "Not Found":
        return {
            "status": "Invalid",
            "employee_id": employee_id,
            "reason": "Employee ID does not exist."
        }

    if employee_result["status"] != "Eligible":
        return {
            "status": "Not Eligible",
            "employee_id": employee_id,
            "reason": "Employee is not eligible for business travel."
        }

    country = employee_result["country"]

    # Policy limits
    if country == "India":
        policy_limit = 2000
        currency = "INR"

    elif country == "US":
        policy_limit = 75
        currency = "USD"

    else:
        return {
            "status": "Invalid",
            "employee_id": employee_id,
            "reason": f"No policy limit configured for {country}."
        }

    # Validate amount
    if amount > policy_limit:
        return {
            "status": "Needs Approval",
            "employee_id": employee_id,
            "country": country,
            "trip_type": trip_type,
            "amount": amount,
            "currency": currency,
            "policy_limit": policy_limit,
            "time": time,
            "reason": (
                f"The trip exceeds the standard {country} "
                f"policy limit of {currency} {policy_limit}."
            )
        }

    return {
        "status": "Approved",
        "employee_id": employee_id,
        "country": country,
        "trip_type": trip_type,
        "amount": amount,
        "currency": currency,
        "policy_limit": policy_limit,
        "time": time,
        "reason": (
            f"The trip is within the standard {country} "
            f"policy limit of {currency} {policy_limit}."
        )
    }
# ============================================================
# TOOL 3: REIMBURSEMENT CALCULATOR
# ============================================================

def calculate_reimbursement(employee_id, amount, expense_type):
    """
    Calculate the reimbursable amount for an employee expense.
    """

    employee_result = check_employee_eligibility(employee_id)

    if employee_result["status"] == "Not Found":
        return {
            "status": "Invalid",
            "employee_id": employee_id,
            "reason": "Employee ID does not exist."
        }

    if employee_result["status"] != "Eligible":
        return {
            "status": "Not Eligible",
            "employee_id": employee_id,
            "reason": "Employee is not eligible for reimbursement."
        }

    # Standard reimbursement rule
    reimbursable_amount = amount

    return {
        "status": "Calculated",
        "employee_id": employee_id,
        "expense_type": expense_type,
        "submitted_amount": amount,
        "reimbursable_amount": reimbursable_amount,
        "currency": "INR" if employee_result["country"] == "India" else "USD",
        "reason": "Expense is eligible for reimbursement."
    }
