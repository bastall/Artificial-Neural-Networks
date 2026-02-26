
# Default Python interpreter
PYTHON = python3

# Default target
all: run

# Run the classifier
run:
	$(PYTHON) classifier.py

# Clean generated files
clean:
	rm -f best_fashion_model.pth final_fashion_model.pth
	rm -f log.txt
	rm -f *.pyc
	rm -rf __pycache__


