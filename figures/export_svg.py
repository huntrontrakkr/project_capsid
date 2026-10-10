"""Export the still figure crossing_pairing.svg from crossing_pairing.html (default state:
shared face at the top of A, the example shells from game_rules.md).

    uv run --no-project --with playwright python export_svg.py
"""
import asyncio
import re
from pathlib import Path

from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
LIGHT = {"--text-primary": "#1f1e1d", "--text-secondary": "#5f5e5a", "--text-muted": "#8b8a85", "--text-accent": "#185FA5",
         "--surface-1": "#f3f2ec", "--surface-2": "#ffffff", "--border-strong": "rgba(0,0,0,.24)",
         "--alike": "#1D9E75", "--differ": "#D85A30", "--page": "#faf9f5"}


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto((HERE / "crossing_pairing.html").as_uri())
        body = await page.evaluate("document.getElementById('g').innerHTML")
        summary = await page.evaluate("document.querySelector('#out .big').textContent")
        await browser.close()
    body = re.sub(r"var\((--[a-z0-9-]+)\)", lambda m: LIGHT[m.group(1)], body)
    body = re.sub(r'<circle [^>]*fill="transparent"/>', "", body)
    font = 'font-family="system-ui,-apple-system,Segoe UI,sans-serif"'
    legend = (f'<line x1="40" y1="372" x2="62" y2="372" stroke="{LIGHT["--alike"]}" stroke-width="2"/>'
              f'<text x="68" y="376" font-size="11" fill="{LIGHT["--text-secondary"]}">alike</text>'
              f'<line x1="110" y1="372" x2="132" y2="372" stroke="{LIGHT["--differ"]}" stroke-width="2"/>'
              f'<text x="138" y="376" font-size="11" fill="{LIGHT["--text-secondary"]}">different</text>'
              f'<line x1="200" y1="372" x2="222" y2="372" stroke="{LIGHT["--text-accent"]}" stroke-width="1.2" stroke-dasharray="5 4"/>'
              f'<text x="228" y="376" font-size="11" fill="{LIGHT["--text-secondary"]}">shared face</text>'
              f'<text x="180" y="400" font-size="12.5" text-anchor="middle" fill="{LIGHT["--text-primary"]}">{summary}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 416" width="360" height="416" {font}>'
           f'<rect width="360" height="416" fill="{LIGHT["--page"]}"/>{body}{legend}</svg>\n')
    (HERE / "crossing_pairing.svg").write_text(svg)
    print("wrote crossing_pairing.svg:", summary)


asyncio.run(main())
