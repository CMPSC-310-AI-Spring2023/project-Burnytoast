# Project Report

## : Andrew Briercheck, Miles Franck, and Nic Ingerson

## Project Summary

For our final project, we are developing a synthetic politician. To do this, we are creating a text generator; taking
speeches from previous American presidents to create a new one. Then, we are going to attempt to do some voice synthesis
to read the speech.

## Motivation

This felt like an enjoyable extention from the previous projects we pursued. Developing a "personality" of sorts seemed like a natural direction to take natural language processing in: seeing what the computer would be able to develop in a sense of "personality" we can understand.

## Background

To develop a sense of "personality" and "beliefs" for this project, we sat down in GFC and put together a document full of "beliefs" we wanted to be reflected in our "speech". After fleshing out a number of beliefs, we then fed it presential speeches to use as a reference, and hoped that it would regurgitate something in a similar vein.

## Project Implementation Details

To run the speech generator:

1. Make sure that python is installed on your system. To check this, run the following command: `python --version`. Otherwise, check the [documentation website](https://www.python.org) to find instructions on how to install python on your system.
2. Next, you need to clone the repository from GitHub. You can do this by opening your command line interface, navigating to the directory where you want to clone the repository, and typing `git clone git@github.com:CMPSC-310-AI-Spring2023/project-Burnytoast.git`
3. Then, you need to install the `openai` package with the following command: `pip install openai`.
4. After installing the `openai` package, navigate to the directory where you cloned the repository. Then, you need to navigate to the `src` directory and open the `text_generator.py` file. Edit the line of code with the comment `Add OpenAI API Key` to contain your OpenAI API key within the `''`.
5. Finally, run the `text_generator.py` file by running the following command: `python3[/python/py] text_generator.py`

To run PoliticanStableDiffusion.ipynb:
1. Open in Colab
2. Ensure the program is utilizing GPU (runtime > change runtime type > GPU)
3. Run cell by cell, in order.

## Testing Details

The StableDiffusion was somewhat uncooperative: we found that relocating the files, transpoting the ipynb into VsCode did not work. After relocating to Colab, the program was cooperating some. The code was then trimmed an altered to fit our needs. The output has been 12 images that vary slightly on the contents: they tend to generally be pictures of generic politicians, sometimes with LGBTQ+ flags in the background, and facial features entering the uncanny valley. 

## Sample Output

```cmd
My fellow citizens, I stand before you today as your newly elected representative. I am honoured and humbled by the trust you have placed in me, and I promise to work hard on your behalf.

In these uncertain times, it is more important than ever that we stand together and support one another. I will do everything in my power to make sure that your voices are heard in government.

My vision for the future is one where we all work together for the common good. I believe that we can achieve great things if we put our differences aside and focus on what we have in common.

I am committed to making your lives better, and I promise to always put your interests first. Thank you for your support, and I look forward to serving you in the years to come.
```

## Experimental Results

When attempting to write the text generation program, we ran into issues with how the program reads certain files and how that affects some variables. In order to overcome these issues, we decided to utilize the OpenAI API library to generate speeches based on a prompt; allowing for a more user-customizable politician in the end.

## Ethical Implications

Using text generation technology to impersonate politicians and spread false information can be highly unethical and damaging to individuals and society. It is important to approach the use of this technology with caution and to make sure we understand the potential consequences before using it

## Team Working Strategy (if worked in a team)

Our team worked together on all aspects of the project. We searched for speeches to use in training, collaborated on coding tasks, and solved problems together.
