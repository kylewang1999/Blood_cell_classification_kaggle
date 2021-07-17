#!/bin/bash
alias kube="kubectl"
alias getpods="kubectl get pods | grep k5wang"
alias getjobs="kubectl get jobs | grep k5wang"
alias get="getpods && getjobs"
alias logs="kubectl logs -l name=k5wang-train"
alias kubelogin="kubectl create -f login.yaml"
alias kubelogout="kubectl delete -f login.yaml"
alias kubeexec="kubectl exec -it k5wang-login bash"
alias exe="kubeexec"
alias p="pwd"
alias weight_path="/k5wang-volume/Blood_cell_classification_kaggle/cluster/eval-EXP-CIFAR-25EPOCHS"
alias test="python test_colab.py --model_path ../cluster/eval-EXP-CIFAR-25EPOCHS/weights.pt"
#/k5wang-volume/Blood_cell_classification_kaggle/cluster