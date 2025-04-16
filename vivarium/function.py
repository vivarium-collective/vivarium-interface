"""
This is an experimental module for creating function-based steps in a process-bigraph simulation.
"""

from process_bigraph import Step

def register_step(input_types, output_types, config_schema):
    def decorator(func):
        class StepFunction(Step):
            def inputs(self):
                return input_types

            def outputs(self):
                return output_types

            def update(self, state):
                return func(state, self.config)

        # # Register the step
        # STEP_REGISTRY[func.__name__] = FunctionStep

        return StepFunction

    return decorator


def run_function_step():
    @register_step(
        input_types=["a", "b"],
        output_types=["c"],
        config_schema={"param": int}
    )
    def my_step(state, config):
        print("Running my_step")
        print("State:", state)
        print("Config:", config)
        return {"c": state["a"] + state["b"]}

    step = my_step
    state = {"a": 1, "b": 2}
    config = {"param": 3}
    step(state, config)


if __name__ == "__main__":
    run_function_step()
