sudo apt update 
sudo apt install unzip -y

if type "pip3" >/dev/null; then
    echo "pip3"
    pip3 install --editable .
else
    echo "pip"
    pip install --editable .
fi
