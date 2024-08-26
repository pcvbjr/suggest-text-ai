# Suggest-Text

Suggest-Text is an AI-powered REST API designed to provide real-time text suggestions to make typing more accessible. The API offers endpoints for word and sentence completion suggestions, to speed up typing, and full response suggestions to conversations captured over audio. The source code consists of two Python services running as Docker containers.

## Prerequisites
Before you begin, ensure that you have the following prerequisites installed on your system:
1. Docker Engine: Docker is used for containerizing the services. You can download Docker Engine from the official Docker website (https://docs.docker.com/engine/install/).
2. Docker Compose: Docker Compose is used for managing the deployment of containers. You can download Docker Compose from the official Docker website (https://docs.docker.com/compose/install/).
3. NVIDIA Container Toolkit: Suggest-Text uses NVIDIA GPUs to accelerate the inference of a speech-to-text AI model. To use NVIDIA GPUs in the containerized services, you will need to have NVIDIA Container Toolkit installed on your system. You can download the NVIDIA Container Toolkit from the official NVIDIA website (https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html).
4. Large Language Model (LLM) OpenAI API Inference Endpoint: Suggest-Text's full response suggestions endpoint requires access to an LLM to generate the responses. The [Environment Configuration](#environment-configuration) step will require setting the endpoint URL and API key. For more information on setting up the LLM endpoint, see [LLM Deployment](#llm-setup) LLM inference endpoint must comply with the following requirements:
    - Compatible with the [OpenAI Python API](https://github.com/openai/openai-python) with both Chat and Completions endpoints (this is a Python package for inference that works with many LLMs; it does not necessitate using OpenAI LLMs.)
    - Must support returning token logprobs
    - Must be network accessible to the device running Suggest-Text


## Environment Configuration

Before launching Suggest-Text, you must create a file called ```.env``` in the parent directory of Suggest-Text (i.e., at the same level as this README.md and compose.yaml).
Copy and paste the following text into the ```.env``` file:
```bash
OPENAI_BASE_URL=
OPENAI_API_KEY=
USER_NAME=
```
Then, assign OPENAI_BASE_URL to the URL used as the LLM inference endpoint, and assign OPENAI_API_KEY as the API key for this endpoint (see [LLM Setup](#llm-setup) for details).

Assign USER_NAME to a name the user of Suggest-Text goes by. This name will be used in prompts to the LLM, so full response suggestions can make use of the user's name.

An example ```.env``` file could be:
```bash
OPENAI_BASE_URL=http://192.0.2.10:8000/v1
OPENAI_API_KEY=myapikey1
USER_NAME=Erin
```

## Launch
Once the the prerequisites are installed on the machine, and the environment configuration is complete, you can launch Suggest-Text. You can either launch all services together using Docker Compose or launching each service individually using Docker Enginer.

### Launch with Docker Compose
From the Suggest-Text parent directory (where this README.md is located), run the command:
```bash
docker compose up
```

### Launch Services Individually with Docker Engine
First, build each image:
```bash
docker build -t suggest-text-core ./src/core
docker build -t suggest-text-whisper-stt ./src/whisper_stt
```

Then, start a container from each image:
```bash
docker run --net host -p 8080:8080 -e WHISPER_API_URL=http://localhost:8081 suggest-text-core
docker run --gpus all -p 8081:8080 -v ./src/user_data:/user_data whisper-stt
```

## Usage

The Suggest-Text API is exposed at http://localhost:8080. 

The application has the following endpoints:

### `/`
This endpoint returns a home page with a basic message.

### `/docs`
This endpoint contains interactive API docs.

### `/set_env`
This endpoint sets environment variables for the API, such as the URL of the OpenAI API service and the OpenAI API key.

### `/suggest`
This endpoint suggests the next single words and characters based on input text using a language model. Below is an example of a POST request to the /suggest endpoint. The command sends a JSON payload with the input text (that has already been typed) to the `/suggest` endpoint. The response will be a JSON object with the suggested words and characters.
```bash
curl -X POST \\
  http://localhost:8080/suggest \\
  -H 'Content-Type: application/json' \\
  -d '{ "text": "input text here" }'
```


### `/full-response`
This endpoint generates full responses to speech files by first transcribing the speech file using the Whisper STT service, then generating 3 different responses. 
> 📘 High-Level Usage Overview
> 
> 1. The other party verbally says a statement that is captured and stored as an audio file. (Accepted file types are mp3, mp4, mpeg , mpga, m4a, wav, and webm). This audio file is saved in ./user_data/
> 2. A POST request to `\full-response` is made, containing a JSON payload with the value of `speech_file` set to the path to the recently created audio file. The value of `conversation_history` contains any previous statements (from the end user or other party) in the conversation. 
> 3. Under the hood, the audio file will be transcibed by the speech-to-text model and then appended to the conversation history. An LLM is used to generate three different adjectives used to describe a wide spectrum of responses (these roughly follow positive, negative and neutral, but are dynamically assigned to allow for adaptability to the conversational context). Then an LLM generates three different responses in the style of the generated adjectives.
> 4. The responses and adjectives used in generating responses are returned in a JSON object.
> Note that Suggest-Text does not store conversation history itself; it is up to programs using Suggest-Text to pass the any conversation history desired to be used as context for response generation.

Below is an example of a curl command that sends a POST request to the /full-response endpoint. The request contains the path to the speech audio file on the client machine - this file is sent to the speech-to-text on the server. The request also contains the conversation history - this is a list of messages comprising the conversation so far, following the [OpenAI API messages format](https://platform.openai.com/docs/api-reference/chat/create#chat-create-messages). The response will be a JSON object with 3 different generated responses. The returned JSON object also contains adjectives describing each response - this will be used in later versions of Suggest-Test to create more personalized responses.

```bash
curl -X POST \\
  http://localhost:8080/full-response \\
  -F "audio_file=@/home/cvanburen/lenovo-psmf-circular-keyboard/suggest-text-ai/src/user_data/fox_and_dog.m4a" \\
  -F 'convo_history=[{"role": "user", "content": "Hello, how are you?"}, {"role": "assistant", "content": "I am doing well, thank you."}]'
```


> 📘 Note 
>
> The end user of Suggest-Text is represented as the `assistant` and the other party in the conversation is represented as the `user` in the conversation history. In the example shown, the other party started the conversation with "Hello, how are you?" the Suggest-Text end user responded with "I am doing well, thank you." This request is made when the other party has responded, with his/her latest response saved as the audio file `speech.wav`. 


## LLM Setup
### Deploy an On-Prem Model
We recommend using the the vLLM library for deploying LLMs on-premises, since it is easy to deploy with Docker, supports many popular open-source LLMs, has the API features required by Suggest-Text, and has state-of-the-art serving throughput for scaling to multiple users.

You can deploy a containerized LLM inference server with vLLM in a single command line. For example, below we show how to launch Meta's Llama 3 8B Instruct model.

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=[INSERT_YOUR_HUGGING_FACE_HUB_TOKEN]" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.4.2 \
    --model meta-llama/Meta-Llama-3-8B-Instruct --api-key "[CHOOSE_A_KEY]"
```

You can see vLLM's full documentation on its official website (https://docs.vllm.ai/en/latest/).

Some models, like Meta's Llama family, require Hugging Face Hub tokens associated with your account. You can create an account and get a token from [HuggingFace](https://huggingface.co/docs/hub/en/security-tokens).

When deploying on-premises, you can choose your own API key. The [Environment Configuration](#environment-configuration) step in Suggest-Text will require you to input this API key as well as the URL for the LLM inference endpoint. The URL will be the port the deployment is exposed on, with `/v1` appended. For example, if we ran the above command on the a machine with the IPv4 address `192.0.2.10`, the URL would be `http://192.0.2.10:8000/v1`.

### Use an OpenAI Model
You can alternatively use a model from OpenAI by using the URL `https://api.openai.com/v1` and an OpenAI API key.

## License

TBD