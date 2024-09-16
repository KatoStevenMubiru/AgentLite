from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from openai import OpenAI
import unify  # Import unify for UnifyLLM
from agentlite.llm.LLMConfig import LLMConfig

OPENAI_CHAT_MODELS = [
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k-0613",
    "gpt-3.5-turbo-16k",
    "gpt-4",
    "gpt-4-0613",
    "gpt-4-turbo",
    "gpt-4-32k",
    "gpt-4-32k-0613",
    "gpt-4-1106-preview",
]
OPENAI_LLM_MODELS = ["text-davinci-003", "text-ada-001"]


class BaseLLM:
    def __init__(self, llm_config: LLMConfig) -> None:
        self.llm_name = llm_config.llm_name
        self.context_len: int = llm_config.context_len
        self.stop: list = llm_config.stop
        self.max_tokens: int = llm_config.max_tokens
        self.temperature: float = llm_config.temperature
        self.end_of_prompt: str = llm_config.end_of_prompt

    def __call__(self, prompt: str) -> str:
        return self.run(prompt)

    def run(self, prompt: str):
        # return str
        raise NotImplementedError


class OpenAIChatLLM(BaseLLM):
    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config=llm_config)
        self.client = OpenAI(api_key=llm_config.api_key)

    def run(self, prompt: str):
        response = self.client.chat.completions.create(
            model=self.llm_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content


class LangchainLLM(BaseLLM):
    def __init__(self, llm_config: LLMConfig):
        from langchain_openai import OpenAI

        super().__init__(llm_config)
        llm = OpenAI(
            model_name=self.llm_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
        )
        human_template = "{prompt}"
        prompt = PromptTemplate(template=human_template, input_variables=["prompt"])
        self.llm_chain = LLMChain(prompt=prompt, llm=llm)

    def run(self, prompt: str):
        return self.llm_chain.run(prompt)


class LangchainChatModel(BaseLLM):
    def __init__(self, llm_config: LLMConfig):
        from langchain_openai import ChatOpenAI

        super().__init__(llm_config)
        llm = ChatOpenAI(
            model_name=self.llm_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            base_url=llm_config.base_url,
            api_key=llm_config.api_key,
        )
        human_template = "{prompt}"
        prompt = PromptTemplate(template=human_template, input_variables=["prompt"])
        self.llm_chain = LLMChain(prompt=prompt, llm=llm)

    def run(self, prompt: str):
        return self.llm_chain.run(prompt)


class UnifyLLM(BaseLLM):
    """LLM class for interacting with UnifyAI."""

    def __init__(self, llm_config: LLMConfig):
        super().__init__(llm_config)
        self.client = unify.Unify(api_key=llm_config.api_key)

        # Extract model and provider from self.llm_name
        parts = self.llm_name.split("@", 1)
        if len(parts) == 2:
            self.model, self.provider = parts
        else:
            raise ValueError(
                "UnifyLLM: Invalid model format. Expected 'model@provider'."
            )

    def run(self, prompt: str):
        """
        Sends a prompt to UnifyAI and returns the LLM's response.

        Args:
            prompt: The text prompt to send to the LLM.

        Returns:
            The LLM's text response.
        """
        response = self.client.generate(
            prompt,
            model=f"{self.model}@{self.provider}",
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.stop,
        )
        return response.text


def get_llm_backend(llm_config: LLMConfig):
    llm_name = llm_config.llm_name

    # Check if the model name is in the format 'model@provider'
    if '@' in llm_name:
        model, provider = llm_name.split('@', 1)
        llm_config.provider = provider
        if provider == "unify":
            return UnifyLLM(llm_config)

    # Fallback to existing behavior
    if llm_name in OPENAI_CHAT_MODELS:
        return LangchainChatModel(llm_config)
    elif llm_name in OPENAI_LLM_MODELS:
        return LangchainLLM(llm_config)
    else:
        # Raise an error if the model is unsupported
        raise ValueError(f"Unsupported model name: {llm_name}")

