# fin_server.py
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Fin")

@mcp.tool()
def calculate_eps(net_income: float, outstanding_shares: int) -> float:
    """Calculate the EPS of the company using net income and outstanding shares."""
    if outstanding_shares == 0:
        raise ValueError("Outstanding shares cannot be zero.")
    return net_income / outstanding_shares

@mcp.tool()
def calculate_operating_profit_margin(operating_income: float, net_sales: float) -> float:
    """Calculate the operating profit margin of the company using operating income and net sales."""
    if net_sales == 0:
        raise ValueError("Net sales cannot be zero.")
    return operating_income / net_sales

if __name__ == "__main__":
    mcp.run(transport="stdio")