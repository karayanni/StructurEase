# StructurEase
StructurEase

## To run the Streamlit application
- Ensure you have Python 3.9 or higher installed locally

- Clone the repo locally
`git clone https://github.com/karayanni/StructurEase.git`

- Navigate to the project directory
`cd StructurEase`

- Create .env file and add OpenAI key
`touch .env`
`open .env`
OPENAI_API_KEY=INSERT-HERE

- Create virtual environment
`python3 -m venv venv`
`source venv/bin/activate`
`pip install -r requirements.txt`
`deactivate` when needed

- Run the application
`streamlit run app_hard_coded.py`

- App should be launched on localhost in browser; try it out
Upload neiss_2023_filtered_unlabeled.csv (found under Evaluation > NEISS data in repo)
Sample classification request: Need to decide if patient was wearing helmet, no helmet or cannot determine.
Qualification classes: Helmet, No Helmet, cannot determine

## Papers
- https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2822296#editorial-comment-tab
- https://arxiv.org/abs/2310.16427