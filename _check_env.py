for name in ["nbclient", "ipykernel", "jupyter_client", "lightgbm", "pandas", "numpy", "scipy", "seaborn"]:
    try:
        m = __import__(name)
        print(f"{name:15s} {getattr(m, '__version__', '?')}")
    except Exception as e:
        print(f"{name:15s} MISSING ({e})")
