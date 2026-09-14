from mcp.server import MCPServer

from src.tools import (
    check_employee_eligibility,
    validate_trip,
    calculate_reimbursement
)

# Create MCP server
mcp = MCPServer("AI Travel Policy Assistant")


@mcp.tool()
def employee_eligibility(employee_id: str):
    """Check whether an employee is eligible for business travel."""
    return check_employee_eligibility(employee_id)


@mcp.tool()
def trip_validation(
    employee_id: str,
    trip_type: str,
    amount: float,
    time: str
):
    """Validate a business trip against employee and travel policies."""
    return validate_trip(
        employee_id,
        trip_type,
        amount,
        time
    )


@mcp.tool()
def reimbursement_calculator(
    employee_id: str,
    amount: float,
    expense_type: str
):
    """Calculate the reimbursable amount for an employee expense."""
    return calculate_reimbursement(
        employee_id,
        amount,
        expense_type
    )


if __name__ == "__main__":
    mcp.run()
