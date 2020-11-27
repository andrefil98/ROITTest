# ROITTest
    Teste de conhecimento
    O IA é baseado em CNN.
    Para ele funcionar será necessário criar na pasta do projeto uma pasta chamada dataset_teste_api e mudar o nome da pasta do seu data set para dataset_treino tendo dentro dela duas pasta que deverão ser chamadas de Cat e Dog. Dentro delas deverá ter as imagens que servirão para treino eu usei 12500 imagens para cada. As imagens deverão ser chamadas apenas por números incrementados a partir do zero.
    Para usar a API utilizei o app postman para criar requisições e enviei através do método POST um parâmetro e o código da imagem em base 64 "img":"código da imagem em base 64 sem o data:image/jpeg;base64,"
    Um site que já dá para converter a imagem sem dar erro é o https://onlinejpgtools.com/convert-jpg-to-base64 
    Endereço do serviço:http://127.0.0.1:5000/pegaimagem
    O modelo.h5 deverá ficar na pasta raiz do projeto.
    O meu modelo tem uma precisão de 50% porém poderá ser criado um novo apenas não tendo esse arquivo em seu projeto o sugerido é criar um modelo.h5 com 50 epocas, para criar com 50 basta substituir no código a variável Epocas = 3 por Epocas = 50.
    Link para o meu modelo.h5 = https://drive.google.com/file/d/1Ayb1x8zcvywvfnhxq1iqF_QEX8a5d1Cg/view?usp=sharing
