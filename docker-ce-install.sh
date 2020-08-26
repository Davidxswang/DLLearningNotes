#!/bin/bash

sudo apt update

sudo apt install \
	apt-transport-https \
	ca-certificates \
	curl \
	gnupg-agent \
	software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo apt-key fingerprint 0EBFCD88

echo "You should see the following:
pub   rsa4096 2017-02-22 [SCEA]
      9DC8 5822 9FC7 DD38 854A  E2D8 8D81 803C 0EBF CD88
uid           [ unknown] Docker Release (CE deb) <docker@docker.com>
sub   rsa4096 2017-02-22 [S]
"

sudo add-apt-repository \
   	"deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   	$(lsb_release -cs) \
   	stable"

echo "Press 'y' to continue"
while : ; do
read -n 1 k <&1
if [[ $k = y ]] ; then
printf "\ncontinue....\n"
break
else
printf "\nstopped\n"
exit ;
fi
done

echo -e "\e[91minstalling docker.....\e[0m"
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io

echo -e "\e[91mdocker installation finished.\e[0m"
echo -e "\e[91myou may want to add your user to docker group by:\n\e[0m\e[93msudo usermod -aG docker \$USER\nnewgrp docker"