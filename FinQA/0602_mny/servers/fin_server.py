# fin_server.py
from mcp.server.fastmcp import FastMCP
from typing import Annotated

mcp = FastMCP("Fin")

@mcp.tool(
    annotations={
        "title": "Calculate Earnings Per Share",
        "description": (
            "Compute the company’s Earnings Per Share (EPS) by dividing net income "
            "by the total number of outstanding shares.\n\n"
            "Inputs:\n"
            "  • net_income (float): Net profit for the period, in reporting currency.\n"
            "  • outstanding_shares (int): Total number of shares outstanding; must be > 0.\n\n"
            "Output:\n"
            "  • float: EPS value, representing profit per share."
        ),
        "readOnlyHint": True,
        "openWorldHint": False,
        "examples": [
            {"net_income": 5000000.0, "outstanding_shares": 1000000}
        ]
    }
)
def calculate_eps(
    net_income: Annotated[float, "Net profit for the period (float). E.g., 5000000.0"],
    outstanding_shares: Annotated[int, "Shares outstanding (int). Must be > 0. E.g., 1000000"]
) -> float:
    """
    Returns:
        float: Earnings Per Share (EPS), calculated as net_income / outstanding_shares.

    Example:
        calculate_eps(5000000.0, 1000000)  # → 5.0
    """
    if outstanding_shares == 0:
        raise ValueError("Outstanding shares cannot be zero.")
    return net_income / outstanding_shares

@mcp.tool(
    annotations={
        "title": "Calculate Operating Profit Margin",
        "description": (
            "Compute the Operating Profit Margin by dividing operating income "
            "by net sales, expressed as a ratio or percentage.\n\n"
            "Inputs:\n"
            "  • operating_income (float): Earnings from core operations before "
            "interest and taxes, in reporting currency.\n"
            "  • net_sales (float): Total revenue from goods or services sold, "
            "in the same currency; must be > 0.\n\n"
            "Output:\n"
            "  • float: Operating Profit Margin (e.g., 0.15 for 15%)."
        ),
        "readOnlyHint": True,
        "openWorldHint": False,
        "examples": [
            {"operating_income": 250000.0, "net_sales": 1000000.0}
        ]
    }
)
def calculate_operating_profit_margin(
    operating_income: Annotated[
        float,
        "Core operating earnings before interest and taxes (float). E.g., 250000.0"
    ],
    net_sales: Annotated[
        float,
        "Total net sales/revenue (float). Must be > 0. E.g., 1_000_000.0"
    ]
) -> float:
    """
    Returns:
        float: Operating Profit Margin = operating_income / net_sales

    Example:
        calculate_operating_profit_margin(250000.0, 1000000.0)  # → 0.25
    """
    if net_sales == 0:
        raise ValueError("Net sales cannot be zero.")
    return operating_income / net_sales

@mcp.tool(
    annotations={
        "title": "Calculate Current Ratio",
        "description": (
            "Calculate the Current Ratio, which indicates a company’s short-term liquidity.\n\n"
            "Inputs:\n"
            "  • current_assets (float): Total current assets.\n"
            "  • current_liabilities (float): Total current liabilities.\n\n"
            "Output:\n"
            "  • float: Current Ratio (e.g., 1.5 means the company has 1.5x current assets per liability)."
        ),
        "readOnlyHint": True,
        "openWorldHint": False,
        "examples": [
            {"current_assets": 3000000.0, "current_liabilities": 1500000.0}
        ]
    }
)
def calculate_current_ratio(
    current_assets: Annotated[float, "Total current assets (float). E.g., 3_000_000.0"],
    current_liabilities: Annotated[float, "Total current liabilities (float). E.g., 1_500_000.0"]
) -> float:
    """
    Returns:
        float: Current Ratio = current_assets / current_liabilities

    Example:
        calculate_current_ratio(3_000_000.0, 1_500_000.0)  # → 2.0
    """
    if current_liabilities == 0:
        raise ValueError("Current liabilities cannot be zero.")
    
    if current_assets / current_liabilities > 100:
        raise ValueError("Current ratio cannot be larger than 100. Check the units.")
    return abs(current_assets / current_liabilities)


@mcp.tool(
    annotations={
        "title": "Calculate Interest Expense to Income Ratio",
        "description": (
            "Compute the ratio of interest expense to interest income.\n\n"
            "Inputs:\n"
            "  • interest_expense (float): Total interest expense incurred.\n"
            "  • interest_income (float): Total interest income earned; must be > 0.\n\n"
            "Output:\n"
            "  • float: Ratio of expense to income (e.g., 1.25 means expense is 125% of income)."
        ),
        "readOnlyHint": True,
        "openWorldHint": False,
        "examples": [
            {"interest_expense": 50000.0, "interest_income": 40000.0}
        ]
    }
)
def calculate_interest_expense_to_income_ratio(
    interest_expense: Annotated[float, "Total interest expense (float). E.g., 50_000.0"],
    interest_income: Annotated[float, "Total interest income (float). E.g., 40_000.0"]
) -> float:
    """
    Returns:
        float: Ratio of interest expense to income = interest_expense / interest_income

    Example:
        calculate_interest_expense_to_income_ratio(50000.0, 40000.0)  # → 1.25
    """
    if interest_income == 0:
        raise ValueError("Interest income cannot be zero.")
    return abs(interest_expense / interest_income)


@mcp.tool(
    annotations={
        "title": "Calculate Lease Payment Percentage",
        "description": (
            "Calculate the percentage of total minimum lease payments due in a specific year.\n\n"
            "Inputs:\n"
            "  • payments_due_in_year (float): Lease payments due in the target year.\n"
            "  • total_minimum_lease_payments (float): Total of all future minimum lease payments; must be > 0.\n\n"
            "Output:\n"
            "  • float: Percentage of lease payments due (e.g., 9.77 means 9.77%)."
        ),
        "readOnlyHint": True,
        "openWorldHint": False,
        "examples": [
            {"payments_due_in_year": 100.0, "total_minimum_lease_payments": 1023.0}
        ]
    }
)
def calculate_lease_payment_percentage(
    payments_due_in_year: Annotated[float, "Lease payments due in a specific year (float). E.g., 100.0"],
    total_minimum_lease_payments: Annotated[float, "Total future minimum lease payments (float). Must be > 0. E.g., 1023.0"]
) -> float:
    """
    Returns:
        float: (payments_due_in_year / total_minimum_lease_payments) * 100

    Example:
        calculate_lease_payment_percentage(100.0, 1023.0)  # → 9.77
    """
    if total_minimum_lease_payments == 0:
        raise ValueError("Total minimum lease payments cannot be zero.")
    return (payments_due_in_year / total_minimum_lease_payments) * 100


@mcp.tool(
    annotations={
        "title": "Calculate Total Assets Change",
        "description": (
            "Compute the change in a company’s total assets over a period.\n\n"
            "Inputs:\n"
            "  • total_assets_start (float): Total assets at the beginning of the period.\n"
            "  • total_assets_end (float): Total assets at the end of the period.\n\n"
            "Output:\n"
            "  • float: Difference in total assets (end - start)."
        ),
        "readOnlyHint": True,
        "openWorldHint": False,
        "examples": [
            {"total_assets_start": 800000.0, "total_assets_end": 850000.0}
        ]
    }
)
def calculate_total_assets_change(
    total_assets_start: Annotated[float, "Total assets at start of period (float). E.g., 800_000.0"],
    total_assets_end: Annotated[float, "Total assets at end of period (float). E.g., 850_000.0"]
) -> float:
    """
    Returns:
        float: Change in total assets = total_assets_end - total_assets_start

    Example:
        calculate_total_assets_change(800000.0, 850000.0)  # → 50000.0
    """
    return total_assets_end - total_assets_start

if __name__ == "__main__":
    mcp.run(transport="stdio")