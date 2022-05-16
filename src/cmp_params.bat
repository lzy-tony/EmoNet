call "C:\Users\liuze\anaconda3\Scripts\activate.bat" pytorch & python main.py -m mlp -c hidden_size
call "C:\Users\liuze\anaconda3\Scripts\activate.bat" pytorch & python main.py -m mlp -c hidden_num
call "C:\Users\liuze\anaconda3\Scripts\activate.bat" pytorch & python main.py -m mlp -c dropout

call "C:\Users\liuze\anaconda3\Scripts\activate.bat" pytorch & python main.py -m cnn -c kernel_num
call "C:\Users\liuze\anaconda3\Scripts\activate.bat" pytorch & python main.py -m cnn -c kernel_size
call "C:\Users\liuze\anaconda3\Scripts\activate.bat" pytorch & python main.py -m cnn -c dropout

call "C:\Users\liuze\anaconda3\Scripts\activate.bat" pytorch & python main.py -m lstm -c dropout