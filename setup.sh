#!/bin/bash

echo "=== Setting up AGI Holographic Memory Project ==="

# Создаем виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Установка зависимостей
pip install --upgrade pip
pip install torch numpy scipy matplotlib tqdm

# Создаем структуру проекта
mkdir -p core experiments results

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To activate environment:"
echo "  source venv/bin/activate"
echo ""
echo "To test the model:"
echo "  python core/holographic_memory.py"
echo ""
echo "To run full experiment:"
echo "  python experiments/train_prototype.py"