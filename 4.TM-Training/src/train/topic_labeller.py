from openai import OpenAI


class TopicLabeller(object):

    def __init__(
        self,
        openai_api_key,
        model='gpt-3.5-turbo',#'gpt-4o-mini',
        temperature=0.1,
        max_tokens=1000,
        frequency_penalty=0.0,
    ) -> None:
        
        try:
            self._client = OpenAI(
                api_key=openai_api_key
        )
        except KeyError:
            raise Exception(
                "Please set the OPENAI_API_KEY environment variable.")

        
        example_1 = ('network, traffic, vehicle, energy, communication, service, deep, reinforcement, sensor, wireless, road, channel, management, node, UAV',
                     'Traffic Management and Autonomous Driving')

        self.parameters = {
            "model": model,
            "messages": [
                {"role": "system", "content": f"You are a helpful assistant trained on the task of labelling chemical descriptions of the topics of a certain topic model. For example, if I give you the chemical description {example_1[0]}, you will give me the label {example_1[1]}. Just answer with the label, no need to write anything else."
                 },
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "frequency_penalty": frequency_penalty
        }

    def set_parameters(self, **kwargs) -> None:
        """Set parameters for the OpenAI model.

        Parameters
        ----------
        **kwargs : dict
            A dictionary of parameters to set.
        """

        for key, value in kwargs.items():
            if key != "messages":
                self.parameters[key] = value

    def update_messages(
        self,
        messages: list
    ) -> None:
        """Update the messages of the OpenAI model, always keeping the first message (i.e., the system role)

        Parameters
        ----------
        messages : list
            A list of messages to update the model with.
        """

        self.parameters["messages"] = [
            self.parameters["messages"][0], *messages]

        return

    def _promt(self, gpt_prompt) -> str:
        """Promt the OpenAI ChatCompletion model with a message.

        Parameters
        ----------
        gpt_prompt : str
            A message to promt the model with.

        Returns
        -------
        str
            The response of the OpenAI model.
        """

        message = [{"role": "user", "content": gpt_prompt}]
        self.update_messages(message)
        response = self._client.chat.completions.create(
            **self.parameters
        )
        return response.choices[0].message.content

    def get_label(self, chem_desc: str) -> str:
        """Get a label for a chemical description.

        Parameters
        ----------
        chem_desc : str
            A chemical description.

        Returns
        -------
        str
            A label for the chemical description.
        """

        gpt_prompt = f"Give me a label for this set of words: {chem_desc}"
        return self._promt(gpt_prompt)

    def get_labels(self, chem_descs: list) -> list:
        """Get labels for a list of chemical descriptions.

        Parameters
        ----------
        chem_descs : list
            A list of chemical descriptions.

        Returns
        -------
        list
            A list of labels for the chemical descriptions.
        """

        def divide_list(input_list, chunk_size):
            return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]
        
        if len(chem_descs) > 10:
            chunked_list = divide_list(chem_descs, 10)

            labels = []
            for labels_chunk in chunked_list:
                gpt_prompt = f"Give me a label for each of the following set of words and return it as a Python list with the labels: {labels_chunk}"
                print(gpt_prompt)
                aux = self._promt(gpt_prompt).replace("'","")
                print(aux)
                aux_eval = eval(aux)
                print(aux_eval)
                labels.extend(aux_eval)
        else:
            
            gpt_prompt = f"Give me a label for each of the following set of words and return it as a Python list with the labels: {chem_descs}"
            labels = eval(self._promt(gpt_prompt))
            
        return labels
