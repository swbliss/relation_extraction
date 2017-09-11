echo 'git'
sshpass -p "ir7753nlp!!" scp -r swjung@143.248.41.54:~/pcnn/.git .
echo 'data_nyt_NRE'
sshpass -p "ir7753nlp!!" scp -r swjung@143.248.41.54:~/pcnn/data_nyt_NRE .
echo 'data_figer'
sshpass -p "ir7753nlp!!" scp -r swjung@143.248.41.54:~/pcnn/data_figer .
echo 'neg_sampling_data'
sshpass -p "ir7753nlp!!" scp -r swjung@143.248.41.54:~/pcnn/neg_sampling_data .
echo 'data_sem_eval'
sshpass -p "ir7753nlp!!" scp -r swjung@143.248.41.54:~/pcnn/data_sem_eval .

sshpass -p "ir7753nlp!!" scp swjung@143.248.41.54:~/pcnn/.gitignore .
sshpass -p "ir7753nlp!!" scp swjung@143.248.41.54:~/pcnn/conv_net_classes.py .
sshpass -p "ir7753nlp!!" scp swjung@143.248.41.54:~/pcnn/data2cv.py .
sshpass -p "ir7753nlp!!" scp swjung@143.248.41.54:~/pcnn/dataset.py .
sshpass -p "ir7753nlp!!" scp swjung@143.248.41.54:~/pcnn/main.py .
sshpass -p "ir7753nlp!!" scp swjung@143.248.41.54:~/pcnn/negsampling.py .
sshpass -p "ir7753nlp!!" scp swjung@143.248.41.54:~/pcnn/nonlinear.py .
sshpass -p "ir7753nlp!!" scp swjung@143.248.41.54:~/pcnn/nyt_ds.py .
