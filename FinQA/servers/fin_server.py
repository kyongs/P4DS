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

if __name__ == "__main__":
    mcp.run(transport="stdio")