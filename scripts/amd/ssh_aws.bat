SET DNS=ec2-54-189-138-199.us-west-2.compute.amazonaws.com
ssh -i "scripts_amd/mmelesse.pem" ubuntu@%DNS%
REM scp -r -i mmelesse.pem ubuntu@%DNS%:~/BERT/bert_train_patch_nvidia .