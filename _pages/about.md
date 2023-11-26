---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>

I'm a PhD student in surgical robotics (2018.9-2024.1) supervised by Siyang Zuo (TJU) and Sophia Bano (UCL).

My research interest includes gastrointetinal image recognition and automatic robotic endoscopy. <a href='https://scholar.google.com/citations?user=L_fC-TMAAAAJ'><img src="https://img.shields.io/endpoint?url={{ url | url_encode }}&logo=Google%20Scholar&labelColor=f6f6f6&color=9cf&style=flat&label=citations"></a>


# 🔥 News
- *2023.10*: &nbsp;🎉🎉 "MonoLoT: Self-Supervised Monocular Depth Estimation in Low-Texture Scenes for Automatic Robotic Endoscopy" submitted to IEEE Journal of Biomedical and Health Informatics (under review). 

# 📝 Publications 

<div class='paper-box'><div class='paper-box-image'><div><img src='images/cibm_2023.jpg' alt="query2_cibm2023" width="300px"></div></div>
<div class='paper-box-text' markdown="1">

**Query2: Query over queries for improving gastrointestinal stromal tumour detection in an endoscopic ultrasound**

**Qi He**, Sophia Bano, Jing Liu, Wentian Liu, Danail Stoyanov, Siyang Zuo. [**code**](https://github.com/howardchina/query2)

*Computers in Biology and Medicine, 2023*


</div>
</div>


# 📖 Educations
- *2018.09 - 2024.01 (now)*, Mechanical Engineering, Tianjin University. 
- *2015.09 - 2018.06*, Computer science, Central South University.
- *2011.09 - 2015.06*, Software Engineering, Central South University. 
