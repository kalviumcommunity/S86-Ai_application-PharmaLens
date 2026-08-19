# PharmaLens Prompt Design

## Chosen Prompt

The clear and constrained prompt was selected as the preferred prompt.

## Why This Prompt Works Better

The prompt clearly defines the task, audience, scope, response length,
and information boundaries.

Compared with the vague prompt, it gives the model fewer decisions to
make about how to structure the answer.

The explicit constraints make the response more focused, consistent,
and easier to use in the PharmaLens application.

This approach will also be useful later in the RAG pipeline, where the
system prompt will instruct the model to answer only from retrieved
document context and refuse unsupported questions.