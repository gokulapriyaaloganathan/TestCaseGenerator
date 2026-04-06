
Step 1 - Create PAT token from ADO.
Step 2 - Copy PAT into `config.json`-> ado_pat
Step 3 - check with concern team and paste secret key under azure_openai_key
Step 4 - Install Python and open terminal.
        Type to create env: python -m venv .venv
        Type to activate env: .\.venv\Scripts\Activate.ps1
        Type to install deps: pip install -r requirements.txt
Step 5 - Run ADO test generator:
        CLI mode: py generate_testcases.py
        UI mode:  py cht.py