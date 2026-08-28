"""CLI for the CRI client intake module."""

import json
import sys
from pathlib import Path

import typer

from cri.intake.parser import parse_excel
from cri.intake.template import generate_template
from cri.intake.validate import validate_company

app = typer.Typer()


@app.command()
def template(
    output: str = typer.Option(
        "data/client_template.xlsx",
        help="Output path for the template Excel file",
    ),
) -> None:
    """Generate a blank client intake template."""
    try:
        generate_template(output)
        typer.echo(f"Template generated successfully: {output}")
    except Exception as e:
        typer.echo(f"Error generating template: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def parse(
    input_file: str = typer.Argument(..., help="Path to the filled-in Excel file"),
) -> None:
    """Parse a filled-in Excel intake file and print a JSON summary."""
    try:
        companies = parse_excel(input_file)

        # Build summary
        summary = {
            "total_companies": len(companies),
            "total_assets": sum(len(c.assets) for c in companies),
            "companies": [],
        }

        for company in companies:
            company_summary = {
                "id": company.id,
                "name": company.name,
                "sector": company.sector,
                "hq_region": company.hq_region,
                "assets": len(company.assets),
                "financials": {
                    "revenue_musd": company.financials.revenue,
                    "ebitda_musd": company.financials.ebitda,
                    "net_debt_musd": company.financials.net_debt,
                    "shares_outstanding_m": company.financials.shares_outstanding,
                    "market_cap_musd": company.financials.market_cap,
                },
                "asset_list": [
                    {
                        "id": asset.id,
                        "name": asset.name,
                        "commodity": asset.commodity.value,
                        "region": asset.region,
                        "baseline_production": asset.baseline_production,
                        "production_unit": asset.production_unit,
                        "carrying_value_musd": asset.carrying_value,
                    }
                    for asset in company.assets
                ],
            }
            summary["companies"].append(company_summary)

        # Print JSON
        print(json.dumps(summary, indent=2))

    except Exception as e:
        typer.echo(f"Error parsing file: {e}", err=True)
        raise typer.Exit(code=1)


@app.command()
def validate(
    input_file: str = typer.Argument(..., help="Path to the filled-in Excel file"),
) -> None:
    """Parse a file and print any data quality warnings."""
    try:
        companies = parse_excel(input_file)

        all_warnings = []
        for company in companies:
            warnings = validate_company(company)
            all_warnings.extend(warnings)

        if not all_warnings:
            typer.echo("No warnings found. Data looks good.")
        else:
            typer.echo(f"Found {len(all_warnings)} warning(s):")
            for warning in all_warnings:
                typer.echo(f"  - {warning}")

    except Exception as e:
        typer.echo(f"Error validating file: {e}", err=True)
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
