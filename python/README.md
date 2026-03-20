## Installation in Virtual Environment

```bash
# 1. Clone repository and navigate to python directory
git clone https://github.com/ail-project/photo-dna.git
cd photo-dna/python

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install maturin
pip install maturin

# 4. Install from source
maturin develop -r

# 5. Explore APIs
python -c "from photo_dna_rs import Hash; help(Hash)"
```
