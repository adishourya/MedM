<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LightRadVLM</title>
    <style>
        @keyframes breathing {
            0% { transform: scale(1); opacity: 0.8; }
            50% { transform: scale(1.05); opacity: 1; }
            100% { transform: scale(1); opacity: 0.8; }
        }

        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f4f4f4;
            margin: 0;
        }

        h1 {
            font-size: 2rem;
            font-weight: bold;
            color: #333;
            animation: breathing 3s infinite ease-in-out;
        }
    </style>
</head>
<body>
    <h1>LightRadVLM<br>Lightweight Radiological VLM</h1>
</body>
</html>


## Code Navigation
![hippo](https://media3.giphy.com/media/aUovxH8Vf9qDu/giphy.gif)

./datasets
  * contains prompting technique for data curation

./experiments
  * contains second stage finetuning techniques (currently only for Medpix and Roco)

./lvlm-interpret
  * (submodule) our diagnostic tool
  * you could run the tool on colab with ./setup_LVLM_INTERPRET.ipynb
