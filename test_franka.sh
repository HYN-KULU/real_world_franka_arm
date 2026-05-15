# from your repo root
cd /home/yinongh/automate/real_world_visual_planning

# activate virtualenv if you have one
# source .venv/bin/activate
# conda activate franka
# ensure editable install (optional if already done)
# pip install -e .

# make the repo root importable for Python (temporary for this terminal)
export PYTHONPATH="$PWD:$PYTHONPATH"

# run the test
python3 frankapanda/calibration/test_franka.py