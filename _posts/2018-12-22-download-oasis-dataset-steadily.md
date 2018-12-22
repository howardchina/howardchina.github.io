---
layout: post
title: Download OASIS dataset steadily on Ubuntu via FTP
date:   2018-12-22 12:02:00 +0800
categories: [brain,dataset]
---

I was supposed to download the OASIS-1 dataset for its fMRI. [Xnat](https://central.xnat.org/app/template/XDATScreen_report_xnat_projectData.vm/search_element/xnat:projectData/search_field/xnat:projectData.ID/search_value/CENTRAL_OASIS_CS) is officially recommend tool for download. However it  didn't work for me. So I tried FTP and it works.

open the [OASIS website](http://www.oasis-brains.org/#access) and read its FTP instructions.

![FTP Download Instructions]({{site.url}}/static/img/posts/OASIS_FTP.png)

**Tools:** lftp, *proxychains* (optinal)

I'll not introduce how to install *proxychains*. I use it because IPV6 is free in TJU, but IPV4 is not. So I build a proxy agent by *proxychains*, than I don't have to pay for the huge network resource.

3 steps to access the OASIS dataset:

1. open a console on Ubuntu, install lftp.

    ```shell
    $ sudo apt-get install lftp
    $ lftp anonymous@ftp.nrg.wustl.edu
    ```

2. then it asked for your Password, just input your e-mail (actually anything is OK).

    connected!!!

    ```sh
    lftp anonymous@ftp.nrg.wustl.edu:/data> ls
    dr-xrwsr-x    3 ftp      ftp            30 Sep 28  2016 data
    dr-xrwsr-x    2 ftp      ftp             7 Jun 18  2010 gsk
    dr-xrwsr-x    2 ftp      ftp             3 Feb 28  2014 private
    dr-xrwsr-x   25 ftp      ftp            40 Sep 05 14:33 pub
    lftp anonymous@ftp.nrg.wustl.edu:/data> cd data
    lftp anonymous@ftp.nrg.wustl.edu:/data> ls
    -r--rw-r--    1 ftp      ftp      10874461059 Mar 18  2009 OAS2_RAW_PART1.tar.gz
    -r--rw-r--    1 ftp      ftp      8449289603 Mar 18  2009 OAS2_RAW_PART2.tar.gz
    -r--rw-r--    1 ftp      ftp      883538165 Jul 21  2008 Vincent_Nature2007_data.tar.gz
    dr-xrwsr-x    2 ftp      ftp            69 Apr 30  2015 hg1000
    -r--rw-r--    1 ftp      ftp      1380527970 Jun 26  2007 oasis_cross-sectional_disc1.tar.gz
    -r--rw-r--    1 ftp      ftp      1294865012 Jun 26  2007 oasis_cross-sectional_disc10.tar.gz
    -r--rw-r--    1 ftp      ftp      1247923584 Jun 26  2007 oasis_cross-sectional_disc11.tar.gz
    -r--rw-r--    1 ftp      ftp      1473055117 Jun 26  2007 oasis_cross-sectional_disc12.tar.gz
    -r--rw-r--    1 ftp      ftp      1403357161 Jun 26  2007 oasis_cross-sectional_disc2.tar.gz
    -r--rw-r--    1 ftp      ftp      1343992875 Jun 26  2007 oasis_cross-sectional_disc3.tar.gz
    -r--rw-r--    1 ftp      ftp      1333582060 Jun 26  2007 oasis_cross-sectional_disc4.tar.gz
    -r--rw-r--    1 ftp      ftp      1362916370 Jun 26  2007 oasis_cross-sectional_disc5.tar.gz
    -r--rw-r--    1 ftp      ftp      1364560560 Jun 26  2007 oasis_cross-sectional_disc6.tar.gz
    -r--rw-r--    1 ftp      ftp      1390914852 Jun 26  2007 oasis_cross-sectional_disc7.tar.gz
    -r--rw-r--    1 ftp      ftp      1332341152 Jun 26  2007 oasis_cross-sectional_disc8.tar.gz
    -r--rw-r--    1 ftp      ftp      1292937504 Jun 26  2007 oasis_cross-sectional_disc9.tar.gz
    -r--rw-r--    1 ftp      ftp      10001233907 Jul 20  2007 oasis_cs_freesurfer_disc1.tar.gz
    -r--rw-r--    1 ftp      ftp      9090665431 Jul 20  2007 oasis_cs_freesurfer_disc10.tar.gz
    -r--rw-r--    1 ftp      ftp      9906158272 Jul 20  2007 oasis_cs_freesurfer_disc11.tar.gz
    -r--rw-r--    1 ftp      ftp      9769478216 Jul 20  2007 oasis_cs_freesurfer_disc2.tar.gz
    -r--rw-r--    1 ftp      ftp      9983477722 Jul 20  2007 oasis_cs_freesurfer_disc3.tar.gz
    -r--rw-r--    1 ftp      ftp      10273223641 Jul 20  2007 oasis_cs_freesurfer_disc4.tar.gz
    -r--rw-r--    1 ftp      ftp      9872006914 Jul 20  2007 oasis_cs_freesurfer_disc5.tar.gz
    -r--rw-r--    1 ftp      ftp      10270081956 Jul 20  2007 oasis_cs_freesurfer_disc6.tar.gz
    -r--rw-r--    1 ftp      ftp      10229874693 Jul 20  2007 oasis_cs_freesurfer_disc7.tar.gz
    -r--rw-r--    1 ftp      ftp      9630920236 Jul 21  2007 oasis_cs_freesurfer_disc8.tar.gz
    -r--rw-r--    1 ftp      ftp      9433239227 Jul 21  2007 oasis_cs_freesurfer_disc9.tar.gz
    -r--rw-r--    1 ftp      ftp      21137786 Sep 28  2016 textspotting.zip
    ```

3. "mget" multiple files with the same prefix.

    ```shell
    lftp anonymous@ftp.nrg.wustl.edu:/data> mget oasis_cross-sectional_disc*
    ```

that's it.