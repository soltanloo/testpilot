import axios from "axios";
import fs from "fs";
import { performance } from "perf_hooks";
import { ICompletionModel } from "./completionModel";
import { trimCompletion } from "./syntax";

const defaultPostOptions = {
  max_tokens: 100, // maximum number of tokens to return
  temperature: 0, // sampling temperature; higher values increase diversity
  n: 1, // number of completions to return
  top_p: 1, // top-p sampling
  stop: null, // optional stopping sequences
};
export type PostOptions = Partial<typeof defaultPostOptions>;

function getEnv(name: string): string {
  const value = process.env[name];
  if (!value) {
    console.error(`Please set the ${name} environment variable.`);
    process.exit(1);
  }
  return value;
}

export class OpenAIModel implements ICompletionModel {
  private readonly apiEndpoint: string;
  private readonly apiKey: string;

  constructor(private readonly instanceOptions: PostOptions = {}) {
    this.apiEndpoint = "https://api.openai.com/v1/chat/completions";
    this.apiKey = getEnv("OPENAI_API_KEY");
    console.log("Using OpenAI GPT-4 API");
  }

  /**
   * Query OpenAI for completions with a given prompt.
   *
   * @param prompt The prompt to use for the completion.
   * @param requestPostOptions The options to use for the request.
   * @returns A promise that resolves to a set of completions.
   */
  public async query(
    prompt: string,
    requestPostOptions: PostOptions = {}
  ): Promise<Set<string>> {
    const headers = {
      "Content-Type": "application/json",
      Authorization: `Bearer ${this.apiKey}`,
    };

    const options = {
      ...defaultPostOptions,
      ...this.instanceOptions,
      ...requestPostOptions,
    };

    performance.mark("openai-query-start");

    const postOptions = {
      model: "gpt-4",
      messages: [{ role: "user", content: prompt }],
      max_tokens: options.max_tokens,
      temperature: options.temperature,
      n: options.n,
      top_p: options.top_p,
      stop: options.stop,
    };

    const res = await axios.post(this.apiEndpoint, postOptions, { headers });

    performance.measure(
      `openai-query:${JSON.stringify({
        ...options,
        promptLength: prompt.length,
      })}`,
      "openai-query-start"
    );

    if (res.status !== 200) {
      throw new Error(
        `Request failed with status ${res.status} and message ${res.statusText}`
      );
    }

    if (!res.data) {
      throw new Error("Response data is empty");
    }

    const completions = new Set<string>();
    for (const choice of res.data.choices || []) {
      completions.add(choice.message.content.trim());
    }

    return completions;
  }

  /**
   * Get completions from OpenAI and postprocess them as needed; print a warning if it did not produce any
   *
   * @param prompt The prompt to use
   * @param temperature Sampling temperature
   */
  public async completions(
    prompt: string,
    temperature: number
  ): Promise<Set<string>> {
    try {
      let result = new Set<string>();
      for (const completion of await this.query(prompt, { temperature })) {
        result.add(trimCompletion(completion));
      }
      return result;
    } catch (err: any) {
      console.warn(`Failed to get completions: ${err.message}`);
      return new Set<string>();
    }
  }
}

if (require.main === module) {
  (async () => {
    const model = new OpenAIModel();
    const prompt = fs.readFileSync(0, "utf8");
    const responses = await model.query(prompt, { n: 1 });
    console.log([...responses][0]);
  })().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
