

# Goals
- have participants run through notebook using google collab to create forecasts of some water variable. Could be streamflow, stream temperature, lake temperature, etc...
- submit forecasts they generate via a form / google drive space. figure out how to do this efficiently. maybe email would suffice with form entry for details.

# things to think about
- not everyone will have google account, can they still run a google collab notebook?
- document how to run the notebook not in collab using uv
- some people will not have great internet connection
- use cronos-2 and tirex2 as the model choices. think about how these models work with different data gaps in target data
- can people upload csv or text file to goolge collab of the timeseries they're interested in to forecast, or do they need to pull from published data somewhere
- could include Zonal stats w/ intersection weighting https://xvec.readthedocs.io/en/stable/zonal_stats.html for people who want to explore further
- Think about adding in simple process-based no parameter model to also make predictions
- point out to participants that https://stac.dynamical.org/catalog.json is very AI friendly
- For non - colab users: uv is simple python install https://github.com/dynamical-org/notebooks/blob/main/README.md
- agreement for being co-author if they submit a forecast?
- should be able to create collab link from github notebook E.g. https://github.com/dynamical-org/notebooks/blob/main/noaa-gefs-forecast-35-day.ipynb
https://colab.research.google.com/github/dynamical-org/notebooks/blob/main/noaa-gefs-forecast-35-day.ipynb

# example notebook
- Copy_of_zero_shot_streamflow_forecast.ipynb is example from Alden (also here
 https://colab.research.google.com/drive/1vxS9vSmwRaRmqgpjgz5lk9pnLZX0LP49)


# eventually
- Do an eval for all the forecasts globally for the final submitted forecasts

# background info and workshop schedule

Day 2: Climate Data -- Tuesday 11 August 2026 Probably 14:00-18:00 UTC
14:00-14:45 UTC - Keynote: The value of open data in environmental sciences: forecasting river flow with open tools alone
Thiago Nascimento (EAWAG)
This talk presents a streamlined workflow for generating a river streamflow forecast for a single catchment using only open data and openly available tools. The approach combines RivRetrieve for accessing river data, delineator for rapid catchment boundary delineation, and dynamical.org for cloud‑optimized weather and climate inputs. Using these datasets, an LSTM rainfall–runoff model is trained with NeuralHydrology and applied to the July 2023 southern Brazil flood, where the resulting streamflow forecast successfully reproduced the observed streamflow peak several days in advance. The talk concludes with a brief discussion of limitations, including reliance on a single weather source, a single‑basin LSTM, and uncertainty derived only from forecast forcing, and encourages attendees to explore the accompanying reproducible notebooks for hands‑on experimentation.
A live stream of the pre-recorded keynote will begin at 14:00 UTC, followed immediately by a live Question and Answer session with the speaker.

14:45-15:00 UTC - Break
15:00-17:00 UTC - Workshop: How to access petabytes of weather forecasts from your laptop
Jake Zwart (USGS and Alden Keefe Sampson (Dynamical)
Accessing large-scale meteorological forecasts is often slow and labor‑intensive, but new fast, interoperable tools are easing this burden. dynamical.org offers open, analysis-ready weather data updated nearly in real time, with a rapidly expanding catalog of global operational models. The platform enables applications such as hydrologic forecasting, algal bloom prediction, and NASA-supported snow‑cover estimation. Ongoing development will broaden support for major global models and emerging AI-based forecasts, helping users apply modern weather data more effectively. In this workshop, we will show participants how to access these data from their laptops.
This workshop will be live.
17:00-18:00 UTC Breakout/Working Groups Sessions
A traditional part of AEMON-J meetings in the past has been to openly discuss ideas for new, joint projects. This time in the schedule is to bring up such ideas, perhaps inspired by the workshops earlier that day, and discuss. If no such ideas are pitched, this time can also be used for small talk, networking, and other social activities.

Advocating Open Data and Open Science in the Aquatic Sciences
The grass-root network DSOS (Data Science and Open Science) and AEMON-J (Aquatic Ecosystem MOdeling Network - Junior) are joining forces this year to organize a 4th 'Virtual Summit: Incorporating Data Science and Open Science in Aquatic Research' that includes fascinating presentations about aquatic research including open data and open science approaches, and multiple days of exciting workshops that include state-of-the-art keynote talks with live hands-on coding exercises. We are adapting the Carpentries Code of Conduct for our meeting.

AEMON-J "Hacking Limnology" Workshop Series
For each day of the "Hacking Limnology" workshop (AEMON-J workshop series), there will be a major theme (e.g., big data, remote sensing, machine learning, and numerical modeling). The general schedule for each day will include a keynote presentation followed by a live Q&A session. The majority of the time will be dedicated to a hands-on coding workshop, where attendees will gain experience in each of the three main themes. Lastly, each day will end with the heart of any AEMON-J meeting: a break-out group format, which will be geared towards spurring new research projects and ideas. Here, we want to engage everybody to find new team mates and initiate collaborations.

DSOS Virtual Summit
For the Virtual Summit: Incorporating Data Science and Open Science in Aquatic Research (DSOS), we will host 10 minute talks from 20 presenters with Live Q&A sessions. Additionally, this year's summit will feature a "Careers in Data Science and Open Science" panel. The summit is intended to bring together diverse, energetic folks who are passionate to share how they bring data science and open science into their research. To get an idea of how the summit will run, you can read about 2020 Virtual Summit, the 2021 Virtual Summit, the 2022 Virtual Summit, the 2023 Virtual Summit, the 2024 Virtual Summit, and the 2025 Virtual Summit, in their respective L&O Bulletin Meeting Highlights pieces.
