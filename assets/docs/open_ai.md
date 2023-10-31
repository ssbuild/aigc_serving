# OpenAI-Compatible RESTful APIs & SDK

FastChat provides OpenAI-compatible APIs for its supported models, so you can use FastChat as a local drop-in replacement for OpenAI APIs.
The FastChat server is compatible with both [openai-python](https://github.com/openai/openai-python) library and cURL commands.

The following OpenAI APIs are supported:
- Chat Completions. (Reference: https://platform.openai.com/docs/api-reference/chat)
- Completions. (Reference: https://platform.openai.com/docs/api-reference/completions)
- Embeddings. (Reference: https://platform.openai.com/docs/api-reference/embeddings)




## Todos
Some features to be implemented:

- [ ] Support more parameters like `logprobs`, `logit_bias`, `user`, `presence_penalty` and `frequency_penalty`
- [ ] Model details (permissions, owner and create time)
- [ ] Edits API
- [ ] Rate Limitation Settings
