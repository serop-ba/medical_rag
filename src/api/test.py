from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
from langchain_community.llms import Ollama
import requests
import time 

def wait_for_service(url):
    timeout = 60  # Timeout in seconds
    start_time = time.time()
    
    while True:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"Service is available at {url}")
                return True
        except requests.ConnectionError:
            pass
        
        if time.time() - start_time > timeout:
            print("Timeout waiting for service to become available")
            return False
        
        time.sleep(1)

def pull_model(model_name: str='llama3', service_name='ollama'):
    service_url = f"http://{service_name}:11434/api/pull"
        
    if wait_for_service(f"http://{service_name}:11434"):
        payload = {
            "name": model_name
        }
        
        try:
            response = requests.post(service_url, json=payload)
            print(f"Curl command executed successfully: {response.status_code}")
            print(response)
        except requests.RequestException as e:
            print(f"Error executing curl command: {e}")
    else:
        raise ValueError("Service unavailable")

def ask_model(model_name: str, prompt: str, service_name="ollama"):
    service_url = f"http://{service_name}:11434/api/generate"
        
    if wait_for_service(f"http://{service_name}:11434"):
        payload = {
            "model": model_name,
            "prompt": prompt
        }
        
        try:
            response = requests.post(service_url, json=payload)
            print(f"Curl command executed successfully: {response.status_code}")
            print(response)
        except requests.RequestException as e:
            print(f"Error executing curl command: {e}")
    else:
        raise ValueError("Service unavailable")
    
def use_langchain():
        # print("build success")
    llm = Ollama(base_url= 'http://ollama:11434', model="llama3")

    llm.invoke("The first man on the moon was ...")







if __name__ == '__main__':
    pull_model('phi3', 'localhost',)
    ask_model("phi3", "The sky is blue", 'localhost')
    # use_langchain()
    



    # print("build success")
    # llm = Ollama(base_url= 'http://host.docker.internal:11434', model="llama2:7b")
    # # llm = Ollama(model="llama2")
#     # llm.invoke("The first man on the moon was ... think step by step")                                
# llm = Ollama(
#     base_url="http://localhost:11434",
#     model="llama3", 
#     callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]))

# llm("hello:")