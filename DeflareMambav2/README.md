# DeflareMambav2: Mamba-Based Nighttime Lens Flare Removal

---

## 📦 Project Structure

```
├── testdata                       
│   ├── Flare7k-real        
│   └── FlareX 
├── weight/                      
│   ├── main.pth        
│   └── FPN.pth
├── mambamain
└── result/                       

├── test.py                       
├── evaluate_Flare7k-real.py              
├── evaluate_FlareX.py       

```

---

### Environment Setup & Installing Mamba

Since this project relies on Mamba, which requires custom CUDA extensions to be compiled locally based on your specific GPU and PyTorch version, please follow the standard installation process below instead of downloading pre-compiled files.

**1. Create a virtual environment and install PyTorch**  
Make sure to install a PyTorch version that matches your system's CUDA version.

```bash  
conda create -n deflare python=3.10 -y  
conda activate deflare  

# (Change to your specific CUDA version)  
conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia -y
```
**2. Install standard dependencies**  

```bash
pip install -r requirements.txt
```

**3. Compile and Install Mamba** 
Ensure that your system has the CUDA Toolkit (NVCC) installed and its version matches your PyTorch CUDA version. Building these packages might take a few minutes.

```bash
cd DeflareMambav2/
pip install packaging
pip install causal-conv1d>=1.2.0
pip install mamba-ssm


```
## 🚀 Quick Testing

### Test Dataset 1: Flare7k-real
```bash
python test.py --input ./testdata/Flare7k-real/input --output ./result/Flare7k-real/
```

### Test Dataset 2: FlareX
```bash
python test.py --input ./testdata/FlareX/input --output ./result/FlareX/
```

Or directly modify paths in `test.py` and run:
```bash
python test.py
```



**Note**: Modify input/output paths directly in `test.py` comments for convenience.

---

## 📊 Evaluation

After running `test.py`, evaluate results:

### Evaluate Flare7k-real Results
```bash
python evaluate_Flare7k-real.py 
```

### Evaluate FlareX Results
```bash
python evaluate_FlareX.py 
```

---

## ⚠️ Important Notes

- **Result Fluctuation**: Results may have minor variations (±0.01) due to randomness in computation
- **Testing Code**: This is the inference version only
- **Training Code**: Full training pipeline will be released after paper acceptance

---

## 📋 Citation


If you use this DeflareMambav2 work, please cite (when published):
```bibtex

```

