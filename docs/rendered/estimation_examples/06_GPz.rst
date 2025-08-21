GPz Estimation Example
======================

**Author:** Sam Schmidt

**Last Run Successfully:** September 26, 2023

A quick demo of running GPz on the typical test data. You should have
installed rail_gpz_v1 (we highly recommend that you do this from within
a custom conda environment so that all dependencies for package versions
are met), either by cloning and installing from github, or with:

::

   pip install pz-rail-gpz-v1

As RAIL is a namespace package, installing rail_gpz_v1 will make
``GPzInformer`` and ``GPzEstimator`` available, and they can be imported
via:

::

   from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

Let’s start with all of our necessary imports:

.. code:: ipython3

    import os
    import matplotlib.pyplot as plt
    import numpy as np
    import rail
    import qp
    from rail.core.data import TableHandle
    from rail.core.stage import RailStage
    from rail.estimation.algos.gpz import GPzInformer, GPzEstimator

.. code:: ipython3

    # set up the DataStore to keep track of data
    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True

.. code:: ipython3

    # find_rail_file is a convenience function that finds a file in the RAIL ecosystem   We have several example data files that are copied with RAIL that we can use for our example run, let's grab those files, one for training/validation, and the other for testing:
    from rail.utils.path_utils import find_rail_file
    trainFile = find_rail_file('examples_data/testdata/test_dc2_training_9816.hdf5')
    testFile = find_rail_file('examples_data/testdata/test_dc2_validation_9816.hdf5')
    training_data = DS.read_file("training_data", TableHandle, trainFile)
    test_data = DS.read_file("test_data", TableHandle, testFile)

Now, we need to set up the stage that will run GPz. We begin by defining
a dictionary with the config options for the algorithm. There are
sensible defaults set, we will override several of these as an example
of how to do this. Config parameters not set in the dictionary will
automatically be set to their default values.

.. code:: ipython3

    gpz_train_dict = dict(n_basis=60, trainfrac=0.8, csl_method="normal", max_iter=150, hdf5_groupname="photometry") 

Let’s set up the training stage. We need to provide a name for the stage
for ceci, as well as a name for the model file that will be written by
the stage. We also include the arguments in the dictionary we wrote
above as additional arguments:

.. code:: ipython3

    # set up the stage to run our GPZ_training
    pz_train = GPzInformer.make_stage(name="GPz_Train", model="GPz_model.pkl", **gpz_train_dict)

We are now ready to run the stage to create the model. We will use the
training data from ``test_dc2_training_9816.hdf5``, which contains
10,225 galaxies drawn from healpix 9816 from the cosmoDC2_v1.1.4
dataset, to train the model. Note that we read this data in called
``train_data`` in the DataStore. Note that we set ``trainfrac`` to 0.8,
so 80% of the data will be used in the “main” training, but 20% will be
reserved by ``GPzInformer`` to determine a SIGMA parameter. We set
``max_iter`` to 150, so we will see 150 steps where the stage tries to
maximize the likelihood. We run the stage as follows:

.. code:: ipython3

    %%time
    pz_train.inform(training_data)


.. parsed-literal::

    Inserting handle into data store.  input: None, GPz_Train
    ngal: 10225
    training model...


.. parsed-literal::

    Iter	 logML/n 		 Train RMSE		 Train RMSE/n		 Valid RMSE		 Valid MLL		 Time    
       1	-3.3874217e-01	 3.1917265e-01	-3.2906491e-01	 3.2563327e-01	[-3.4188040e-01]	 4.5700502e-01


.. parsed-literal::

       2	-2.6769850e-01	 3.0826967e-01	-2.4316866e-01	 3.1458378e-01	[-2.6335723e-01]	 2.2789812e-01


.. parsed-literal::

       3	-2.2257648e-01	 2.8708382e-01	-1.7891653e-01	 2.9286919e-01	[-2.0612655e-01]	 2.8512168e-01


.. parsed-literal::

       4	-1.9634907e-01	 2.6404013e-01	-1.5571036e-01	 2.7034749e-01	[-1.9479383e-01]	 2.0404720e-01


.. parsed-literal::

       5	-1.0068636e-01	 2.5626239e-01	-6.4405566e-02	 2.6031731e-01	[-8.5591003e-02]	 2.0196509e-01
       6	-6.8586284e-02	 2.5105265e-01	-3.7298448e-02	 2.5649354e-01	[-5.8749654e-02]	 2.0566750e-01


.. parsed-literal::

       7	-4.8066273e-02	 2.4774235e-01	-2.3859678e-02	 2.5338412e-01	[-4.6677506e-02]	 2.0579910e-01


.. parsed-literal::

       8	-3.7332729e-02	 2.4603415e-01	-1.6630401e-02	 2.5173157e-01	[-4.0334636e-02]	 2.0681810e-01


.. parsed-literal::

       9	-2.4157042e-02	 2.4363218e-01	-6.3633968e-03	 2.4933168e-01	[-3.0237934e-02]	 2.0764661e-01


.. parsed-literal::

      10	-1.2260425e-02	 2.4118089e-01	 3.5685456e-03	 2.4656433e-01	[-2.0751951e-02]	 2.0429492e-01


.. parsed-literal::

      11	-7.8637403e-03	 2.4051749e-01	 6.3932760e-03	 2.4481173e-01	[-1.0827430e-02]	 2.0374417e-01


.. parsed-literal::

      12	-3.3769993e-03	 2.3990756e-01	 1.0629991e-02	 2.4465669e-01	[-9.6331617e-03]	 2.0970440e-01


.. parsed-literal::

      13	-7.3093233e-04	 2.3937861e-01	 1.3140067e-02	 2.4438584e-01	[-8.7061007e-03]	 2.0636034e-01


.. parsed-literal::

      14	 3.3145288e-03	 2.3854454e-01	 1.7481447e-02	 2.4392140e-01	[-6.1592742e-03]	 2.0845103e-01


.. parsed-literal::

      15	 1.0985650e-01	 2.2676961e-01	 1.3036972e-01	 2.3384430e-01	[ 1.0198508e-01]	 3.2114673e-01


.. parsed-literal::

      16	 1.3882561e-01	 2.2349803e-01	 1.6125195e-01	 2.2975881e-01	[ 1.4416421e-01]	 3.1409645e-01


.. parsed-literal::

      17	 1.8946298e-01	 2.1997008e-01	 2.1206388e-01	 2.2549748e-01	[ 1.9732558e-01]	 2.1505284e-01


.. parsed-literal::

      18	 3.0981421e-01	 2.1550678e-01	 3.3894514e-01	 2.2328653e-01	[ 3.0866355e-01]	 2.1324944e-01


.. parsed-literal::

      19	 3.5015142e-01	 2.1580697e-01	 3.8121091e-01	 2.2376155e-01	[ 3.4931387e-01]	 2.0837498e-01


.. parsed-literal::

      20	 3.9158617e-01	 2.1446481e-01	 4.2290924e-01	 2.2554017e-01	[ 3.8601803e-01]	 2.0416498e-01


.. parsed-literal::

      21	 4.4286005e-01	 2.1165057e-01	 4.7538816e-01	 2.2251910e-01	[ 4.4083977e-01]	 2.0731282e-01


.. parsed-literal::

      22	 5.3205512e-01	 2.1118392e-01	 5.6778014e-01	 2.1706281e-01	[ 5.3645502e-01]	 2.1198845e-01


.. parsed-literal::

      23	 5.7649882e-01	 2.1040850e-01	 6.1559026e-01	 2.1899217e-01	[ 5.5182024e-01]	 2.1677804e-01
      24	 6.1345420e-01	 2.0688384e-01	 6.5226831e-01	 2.1569420e-01	[ 5.9623167e-01]	 1.9683599e-01


.. parsed-literal::

      25	 6.4278440e-01	 2.0468133e-01	 6.8102950e-01	 2.1389276e-01	[ 6.2401633e-01]	 1.8406487e-01
      26	 6.9285860e-01	 2.0238500e-01	 7.2987503e-01	 2.1303976e-01	[ 6.8483824e-01]	 1.8388033e-01


.. parsed-literal::

      27	 7.3442250e-01	 1.9666721e-01	 7.6999886e-01	 2.0789168e-01	[ 7.2014700e-01]	 2.0698142e-01
      28	 7.6972323e-01	 1.9407465e-01	 8.0871431e-01	 2.0576705e-01	[ 7.6441876e-01]	 1.9763708e-01


.. parsed-literal::

      29	 7.9943007e-01	 1.9254779e-01	 8.3922253e-01	 2.0302209e-01	[ 7.9252979e-01]	 1.8375516e-01


.. parsed-literal::

      30	 8.2665683e-01	 1.9204717e-01	 8.6641721e-01	 2.0388406e-01	[ 8.1341646e-01]	 2.1202588e-01


.. parsed-literal::

      31	 8.5004744e-01	 1.9231337e-01	 8.9091725e-01	 2.0483167e-01	[ 8.3689768e-01]	 2.1067548e-01


.. parsed-literal::

      32	 8.7361613e-01	 1.9264263e-01	 9.1498275e-01	 2.0486433e-01	[ 8.5877965e-01]	 2.0712614e-01
      33	 8.9884746e-01	 1.9380521e-01	 9.4083655e-01	 2.0591160e-01	[ 8.8283906e-01]	 2.0009995e-01


.. parsed-literal::

      34	 9.2562985e-01	 1.9163916e-01	 9.6870629e-01	 2.0294867e-01	[ 9.0742993e-01]	 2.0443606e-01


.. parsed-literal::

      35	 9.4621582e-01	 1.9216892e-01	 9.9053120e-01	 2.0311058e-01	[ 9.1665585e-01]	 2.1178293e-01
      36	 9.6461850e-01	 1.8961326e-01	 1.0094903e+00	 1.9974587e-01	[ 9.3972445e-01]	 1.9308901e-01


.. parsed-literal::

      37	 9.7788203e-01	 1.8864568e-01	 1.0231982e+00	 1.9830293e-01	[ 9.5084515e-01]	 2.1124649e-01
      38	 9.9252213e-01	 1.8615730e-01	 1.0386955e+00	 1.9576452e-01	[ 9.5716643e-01]	 1.9798636e-01


.. parsed-literal::

      39	 1.0049749e+00	 1.8537322e-01	 1.0515642e+00	 1.9502028e-01	[ 9.5942867e-01]	 2.0276070e-01


.. parsed-literal::

      40	 1.0173370e+00	 1.8413469e-01	 1.0639632e+00	 1.9411262e-01	[ 9.7320447e-01]	 2.1072793e-01
      41	 1.0330735e+00	 1.8188180e-01	 1.0797597e+00	 1.9232680e-01	[ 9.9409843e-01]	 1.7368984e-01


.. parsed-literal::

      42	 1.0452032e+00	 1.7982310e-01	 1.0918538e+00	 1.9018622e-01	[ 1.0099003e+00]	 1.8494964e-01


.. parsed-literal::

      43	 1.0591074e+00	 1.7841531e-01	 1.1053743e+00	 1.8813599e-01	[ 1.0233280e+00]	 2.1284127e-01


.. parsed-literal::

      44	 1.0698850e+00	 1.7720777e-01	 1.1166184e+00	 1.8628463e-01	[ 1.0312252e+00]	 2.0624733e-01


.. parsed-literal::

      45	 1.0844824e+00	 1.7600786e-01	 1.1315937e+00	 1.8468868e-01	[ 1.0479008e+00]	 2.1481919e-01
      46	 1.0941519e+00	 1.7382294e-01	 1.1429412e+00	 1.8264595e-01	[ 1.0571309e+00]	 1.7695165e-01


.. parsed-literal::

      47	 1.1095549e+00	 1.7387455e-01	 1.1579149e+00	 1.8311957e-01	[ 1.0786487e+00]	 1.9852018e-01


.. parsed-literal::

      48	 1.1152649e+00	 1.7381689e-01	 1.1632237e+00	 1.8337366e-01	[ 1.0846244e+00]	 2.1439838e-01


.. parsed-literal::

      49	 1.1264494e+00	 1.7213095e-01	 1.1746565e+00	 1.8205379e-01	[ 1.0931567e+00]	 2.0731378e-01
      50	 1.1357770e+00	 1.7023080e-01	 1.1845655e+00	 1.8069953e-01	[ 1.1010508e+00]	 1.9639564e-01


.. parsed-literal::

      51	 1.1439433e+00	 1.6910348e-01	 1.1929038e+00	 1.7949318e-01	[ 1.1076494e+00]	 2.0669794e-01


.. parsed-literal::

      52	 1.1543767e+00	 1.6739960e-01	 1.2033519e+00	 1.7750175e-01	[ 1.1192584e+00]	 2.0421982e-01
      53	 1.1679838e+00	 1.6538592e-01	 1.2176014e+00	 1.7538668e-01	[ 1.1298542e+00]	 1.7910695e-01


.. parsed-literal::

      54	 1.1795782e+00	 1.6264611e-01	 1.2291675e+00	 1.7246172e-01	[ 1.1473447e+00]	 2.0815539e-01


.. parsed-literal::

      55	 1.1877382e+00	 1.6149171e-01	 1.2372475e+00	 1.7184829e-01	[ 1.1549254e+00]	 2.0567012e-01
      56	 1.1993874e+00	 1.5950220e-01	 1.2493060e+00	 1.7056736e-01	[ 1.1658010e+00]	 1.7143106e-01


.. parsed-literal::

      57	 1.2086293e+00	 1.5754902e-01	 1.2592718e+00	 1.6991898e-01	[ 1.1766489e+00]	 2.0662165e-01
      58	 1.2183524e+00	 1.5545224e-01	 1.2689557e+00	 1.6828565e-01	[ 1.1896322e+00]	 1.9846344e-01


.. parsed-literal::

      59	 1.2247072e+00	 1.5497673e-01	 1.2751772e+00	 1.6790562e-01	[ 1.1964786e+00]	 1.6381288e-01


.. parsed-literal::

      60	 1.2330948e+00	 1.5391947e-01	 1.2837113e+00	 1.6737809e-01	[ 1.2060434e+00]	 2.0907903e-01


.. parsed-literal::

      61	 1.2396958e+00	 1.5306020e-01	 1.2907504e+00	 1.6796632e-01	  1.2051221e+00 	 2.1276283e-01
      62	 1.2486771e+00	 1.5271982e-01	 1.2996544e+00	 1.6769080e-01	[ 1.2172740e+00]	 1.8698645e-01


.. parsed-literal::

      63	 1.2542334e+00	 1.5257840e-01	 1.3052469e+00	 1.6758680e-01	[ 1.2209966e+00]	 2.1567512e-01


.. parsed-literal::

      64	 1.2629544e+00	 1.5193693e-01	 1.3141867e+00	 1.6720726e-01	[ 1.2244019e+00]	 2.1374464e-01


.. parsed-literal::

      65	 1.2713597e+00	 1.5092177e-01	 1.3227321e+00	 1.6653419e-01	[ 1.2344938e+00]	 2.1371317e-01
      66	 1.2787157e+00	 1.4985131e-01	 1.3302690e+00	 1.6587292e-01	[ 1.2368242e+00]	 1.8865108e-01


.. parsed-literal::

      67	 1.2828195e+00	 1.4943659e-01	 1.3343387e+00	 1.6552536e-01	[ 1.2407447e+00]	 2.1331382e-01
      68	 1.2904534e+00	 1.4833691e-01	 1.3421793e+00	 1.6467339e-01	[ 1.2495733e+00]	 1.8343854e-01


.. parsed-literal::

      69	 1.2968135e+00	 1.4870495e-01	 1.3487918e+00	 1.6525946e-01	[ 1.2537271e+00]	 1.9866705e-01


.. parsed-literal::

      70	 1.3032061e+00	 1.4803097e-01	 1.3551909e+00	 1.6435945e-01	[ 1.2633176e+00]	 2.1985626e-01


.. parsed-literal::

      71	 1.3077728e+00	 1.4748419e-01	 1.3598546e+00	 1.6394067e-01	[ 1.2686014e+00]	 2.1302843e-01
      72	 1.3126650e+00	 1.4723057e-01	 1.3648922e+00	 1.6387563e-01	[ 1.2737101e+00]	 1.8586779e-01


.. parsed-literal::

      73	 1.3188703e+00	 1.4644963e-01	 1.3714239e+00	 1.6386918e-01	[ 1.2775522e+00]	 2.0963883e-01


.. parsed-literal::

      74	 1.3247727e+00	 1.4620020e-01	 1.3774975e+00	 1.6409947e-01	[ 1.2802749e+00]	 2.0899343e-01


.. parsed-literal::

      75	 1.3308696e+00	 1.4602894e-01	 1.3837335e+00	 1.6427035e-01	[ 1.2839615e+00]	 2.0200467e-01
      76	 1.3366199e+00	 1.4570381e-01	 1.3896066e+00	 1.6417938e-01	[ 1.2880951e+00]	 1.9812727e-01


.. parsed-literal::

      77	 1.3437436e+00	 1.4499332e-01	 1.3967876e+00	 1.6354197e-01	[ 1.2948261e+00]	 1.9691849e-01


.. parsed-literal::

      78	 1.3495541e+00	 1.4357964e-01	 1.4027132e+00	 1.6214846e-01	[ 1.2992649e+00]	 2.0645952e-01


.. parsed-literal::

      79	 1.3546177e+00	 1.4290039e-01	 1.4076505e+00	 1.6181882e-01	[ 1.3007063e+00]	 2.1852231e-01


.. parsed-literal::

      80	 1.3589173e+00	 1.4212851e-01	 1.4119402e+00	 1.6132319e-01	[ 1.3022442e+00]	 2.0913482e-01


.. parsed-literal::

      81	 1.3646682e+00	 1.4116248e-01	 1.4178425e+00	 1.6094150e-01	  1.3011532e+00 	 2.1059728e-01


.. parsed-literal::

      82	 1.3672927e+00	 1.4059789e-01	 1.4206332e+00	 1.6047323e-01	  1.3018795e+00 	 2.0823097e-01


.. parsed-literal::

      83	 1.3702611e+00	 1.4057837e-01	 1.4235052e+00	 1.6033727e-01	[ 1.3046310e+00]	 2.1208239e-01


.. parsed-literal::

      84	 1.3746845e+00	 1.4008062e-01	 1.4279667e+00	 1.5956265e-01	[ 1.3079770e+00]	 2.0098639e-01
      85	 1.3782195e+00	 1.3955663e-01	 1.4315800e+00	 1.5879356e-01	[ 1.3104664e+00]	 2.0823741e-01


.. parsed-literal::

      86	 1.3798091e+00	 1.3764375e-01	 1.4336655e+00	 1.5634843e-01	  1.3057474e+00 	 2.0048952e-01
      87	 1.3878605e+00	 1.3757386e-01	 1.4415156e+00	 1.5614657e-01	[ 1.3154292e+00]	 1.7808723e-01


.. parsed-literal::

      88	 1.3898679e+00	 1.3742874e-01	 1.4435162e+00	 1.5608291e-01	[ 1.3172481e+00]	 2.0334077e-01
      89	 1.3932364e+00	 1.3699197e-01	 1.4470364e+00	 1.5563253e-01	[ 1.3185090e+00]	 1.9918537e-01


.. parsed-literal::

      90	 1.3976167e+00	 1.3652415e-01	 1.4515705e+00	 1.5511619e-01	  1.3178251e+00 	 1.8853164e-01


.. parsed-literal::

      91	 1.3998549e+00	 1.3606605e-01	 1.4540791e+00	 1.5440923e-01	  1.3184085e+00 	 2.1537995e-01
      92	 1.4048888e+00	 1.3584168e-01	 1.4589366e+00	 1.5416525e-01	[ 1.3190726e+00]	 1.9700265e-01


.. parsed-literal::

      93	 1.4069358e+00	 1.3574267e-01	 1.4609054e+00	 1.5402322e-01	[ 1.3194014e+00]	 3.8416600e-01


.. parsed-literal::

      94	 1.4101235e+00	 1.3544198e-01	 1.4640724e+00	 1.5367274e-01	[ 1.3194877e+00]	 2.1131396e-01


.. parsed-literal::

      95	 1.4159251e+00	 1.3483583e-01	 1.4699249e+00	 1.5309380e-01	  1.3189640e+00 	 2.0975304e-01


.. parsed-literal::

      96	 1.4195772e+00	 1.3419833e-01	 1.4736029e+00	 1.5254995e-01	  1.3180518e+00 	 3.2685614e-01


.. parsed-literal::

      97	 1.4233904e+00	 1.3386181e-01	 1.4774511e+00	 1.5240252e-01	  1.3176346e+00 	 2.0796990e-01


.. parsed-literal::

      98	 1.4270424e+00	 1.3359496e-01	 1.4811793e+00	 1.5246538e-01	  1.3187440e+00 	 2.0844436e-01


.. parsed-literal::

      99	 1.4301436e+00	 1.3349453e-01	 1.4843603e+00	 1.5266826e-01	  1.3177019e+00 	 2.0300508e-01
     100	 1.4334923e+00	 1.3325729e-01	 1.4877785e+00	 1.5263538e-01	  1.3184509e+00 	 1.8384981e-01


.. parsed-literal::

     101	 1.4361000e+00	 1.3306035e-01	 1.4903953e+00	 1.5244082e-01	[ 1.3198441e+00]	 2.1187067e-01


.. parsed-literal::

     102	 1.4388136e+00	 1.3281740e-01	 1.4931384e+00	 1.5217783e-01	  1.3192042e+00 	 2.1691155e-01
     103	 1.4427458e+00	 1.3260685e-01	 1.4971401e+00	 1.5182360e-01	[ 1.3209536e+00]	 1.9757533e-01


.. parsed-literal::

     104	 1.4460149e+00	 1.3229620e-01	 1.5005613e+00	 1.5158667e-01	  1.3153551e+00 	 1.8394065e-01


.. parsed-literal::

     105	 1.4489106e+00	 1.3223968e-01	 1.5034449e+00	 1.5146768e-01	  1.3182951e+00 	 2.1758580e-01
     106	 1.4515602e+00	 1.3213864e-01	 1.5061356e+00	 1.5127441e-01	  1.3196420e+00 	 1.9848609e-01


.. parsed-literal::

     107	 1.4546764e+00	 1.3189533e-01	 1.5093213e+00	 1.5078863e-01	  1.3198942e+00 	 2.0599461e-01


.. parsed-literal::

     108	 1.4581479e+00	 1.3105608e-01	 1.5130352e+00	 1.4948026e-01	  1.3177595e+00 	 2.0373988e-01
     109	 1.4617438e+00	 1.3088388e-01	 1.5166007e+00	 1.4913661e-01	  1.3179955e+00 	 1.9978166e-01


.. parsed-literal::

     110	 1.4633776e+00	 1.3083933e-01	 1.5181617e+00	 1.4908189e-01	  1.3199006e+00 	 2.1713495e-01


.. parsed-literal::

     111	 1.4666815e+00	 1.3056553e-01	 1.5215557e+00	 1.4868202e-01	  1.3197983e+00 	 2.0817709e-01
     112	 1.4703659e+00	 1.3048271e-01	 1.5255108e+00	 1.4823467e-01	  1.3150698e+00 	 1.7456341e-01


.. parsed-literal::

     113	 1.4737222e+00	 1.3012015e-01	 1.5291114e+00	 1.4758509e-01	  1.3121687e+00 	 1.9691014e-01
     114	 1.4765091e+00	 1.2998270e-01	 1.5320091e+00	 1.4737731e-01	  1.3101934e+00 	 1.9631577e-01


.. parsed-literal::

     115	 1.4789178e+00	 1.3002741e-01	 1.5344479e+00	 1.4741815e-01	  1.3098885e+00 	 2.0249605e-01


.. parsed-literal::

     116	 1.4812475e+00	 1.2996384e-01	 1.5368091e+00	 1.4745591e-01	  1.3064395e+00 	 2.1807909e-01


.. parsed-literal::

     117	 1.4844162e+00	 1.2987350e-01	 1.5400911e+00	 1.4766957e-01	  1.3029035e+00 	 2.1119523e-01
     118	 1.4868970e+00	 1.2975623e-01	 1.5425791e+00	 1.4775850e-01	  1.2995334e+00 	 1.9281292e-01


.. parsed-literal::

     119	 1.4891294e+00	 1.2967163e-01	 1.5448226e+00	 1.4777587e-01	  1.2985265e+00 	 2.1642351e-01


.. parsed-literal::

     120	 1.4925625e+00	 1.2946475e-01	 1.5482808e+00	 1.4794405e-01	  1.2952979e+00 	 2.1972704e-01


.. parsed-literal::

     121	 1.4946311e+00	 1.2947004e-01	 1.5504431e+00	 1.4819884e-01	  1.2857562e+00 	 2.1037078e-01


.. parsed-literal::

     122	 1.4967335e+00	 1.2938424e-01	 1.5524124e+00	 1.4808539e-01	  1.2896137e+00 	 2.1608472e-01


.. parsed-literal::

     123	 1.4987355e+00	 1.2930809e-01	 1.5544348e+00	 1.4803931e-01	  1.2858003e+00 	 2.1326351e-01


.. parsed-literal::

     124	 1.5007537e+00	 1.2927036e-01	 1.5564529e+00	 1.4809885e-01	  1.2806878e+00 	 2.1088743e-01
     125	 1.5032567e+00	 1.2899580e-01	 1.5591120e+00	 1.4788345e-01	  1.2689034e+00 	 1.8537068e-01


.. parsed-literal::

     126	 1.5058025e+00	 1.2885142e-01	 1.5616323e+00	 1.4786624e-01	  1.2687608e+00 	 2.1607780e-01


.. parsed-literal::

     127	 1.5073340e+00	 1.2875254e-01	 1.5631740e+00	 1.4778259e-01	  1.2681018e+00 	 2.1531940e-01


.. parsed-literal::

     128	 1.5096038e+00	 1.2852416e-01	 1.5655418e+00	 1.4751927e-01	  1.2667416e+00 	 2.2003913e-01


.. parsed-literal::

     129	 1.5119246e+00	 1.2820637e-01	 1.5680681e+00	 1.4721468e-01	  1.2555476e+00 	 2.0771790e-01


.. parsed-literal::

     130	 1.5142914e+00	 1.2801451e-01	 1.5704829e+00	 1.4701830e-01	  1.2532234e+00 	 2.0355654e-01
     131	 1.5162897e+00	 1.2785583e-01	 1.5725043e+00	 1.4684547e-01	  1.2547166e+00 	 1.9556689e-01


.. parsed-literal::

     132	 1.5181564e+00	 1.2777022e-01	 1.5743970e+00	 1.4681820e-01	  1.2528619e+00 	 1.8796563e-01
     133	 1.5203475e+00	 1.2747583e-01	 1.5766861e+00	 1.4645972e-01	  1.2557897e+00 	 1.9754195e-01


.. parsed-literal::

     134	 1.5221655e+00	 1.2734145e-01	 1.5784712e+00	 1.4633793e-01	  1.2555592e+00 	 2.1366525e-01


.. parsed-literal::

     135	 1.5239634e+00	 1.2709271e-01	 1.5802993e+00	 1.4608710e-01	  1.2492534e+00 	 2.1986914e-01
     136	 1.5254838e+00	 1.2679508e-01	 1.5818297e+00	 1.4574346e-01	  1.2448141e+00 	 1.9926262e-01


.. parsed-literal::

     137	 1.5272799e+00	 1.2647747e-01	 1.5836989e+00	 1.4538010e-01	  1.2417524e+00 	 2.1056867e-01
     138	 1.5288140e+00	 1.2611255e-01	 1.5853246e+00	 1.4496770e-01	  1.2322845e+00 	 1.9031048e-01


.. parsed-literal::

     139	 1.5301214e+00	 1.2599881e-01	 1.5866553e+00	 1.4492756e-01	  1.2326956e+00 	 1.8357372e-01
     140	 1.5317627e+00	 1.2582928e-01	 1.5883711e+00	 1.4478394e-01	  1.2341370e+00 	 1.8232346e-01


.. parsed-literal::

     141	 1.5332577e+00	 1.2560799e-01	 1.5899269e+00	 1.4463798e-01	  1.2344233e+00 	 1.9246292e-01


.. parsed-literal::

     142	 1.5350768e+00	 1.2546874e-01	 1.5917919e+00	 1.4449384e-01	  1.2343585e+00 	 2.0905638e-01
     143	 1.5360889e+00	 1.2538207e-01	 1.5928027e+00	 1.4439452e-01	  1.2340280e+00 	 1.8508911e-01


.. parsed-literal::

     144	 1.5373022e+00	 1.2528134e-01	 1.5940118e+00	 1.4441305e-01	  1.2310377e+00 	 2.1631861e-01


.. parsed-literal::

     145	 1.5388233e+00	 1.2506367e-01	 1.5955997e+00	 1.4436202e-01	  1.2348494e+00 	 2.0325732e-01


.. parsed-literal::

     146	 1.5404445e+00	 1.2495154e-01	 1.5972046e+00	 1.4456030e-01	  1.2311420e+00 	 2.8114080e-01


.. parsed-literal::

     147	 1.5413878e+00	 1.2493547e-01	 1.5981428e+00	 1.4464326e-01	  1.2314799e+00 	 2.1225953e-01
     148	 1.5429263e+00	 1.2481468e-01	 1.5997168e+00	 1.4464074e-01	  1.2371942e+00 	 2.0000672e-01


.. parsed-literal::

     149	 1.5440440e+00	 1.2480419e-01	 1.6009016e+00	 1.4486592e-01	  1.2323649e+00 	 2.1166205e-01


.. parsed-literal::

     150	 1.5453342e+00	 1.2466965e-01	 1.6022084e+00	 1.4476723e-01	  1.2320127e+00 	 2.0753789e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.06 s, total: 2min 5s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f3888d4a590>



This should have taken about 30 seconds on a typical desktop computer,
and you should now see a file called ``GPz_model.pkl`` in the directory.
This model file is used by the ``GPzEstimator`` stage to determine our
redshift PDFs for the test set of galaxies. Let’s set up that stage,
again defining a dictionary of variables for the config params:

.. code:: ipython3

    gpz_test_dict = dict(hdf5_groupname="photometry", model="GPz_model.pkl")
    
    gpz_run = GPzEstimator.make_stage(name="gpz_run", **gpz_test_dict)

Let’s run the stage and compute photo-z’s for our test set:

.. code:: ipython3

    %%time
    results = gpz_run.estimate(test_data)


.. parsed-literal::

    Inserting handle into data store.  model: GPz_model.pkl, gpz_run
    Process 0 running estimator on chunk 0 - 10,000
    Process 0 estimating GPz PZ PDF for rows 0 - 10,000


.. parsed-literal::

    Inserting handle into data store.  output_gpz_run: inprogress_output_gpz_run.hdf5, gpz_run


.. parsed-literal::

    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.14 s, sys: 46 ms, total: 2.19 s
    Wall time: 671 ms


This should be very fast, under a second for our 20,449 galaxies in the
test set. Now, let’s plot a scatter plot of the point estimates, as well
as a few example PDFs. We can get access to the ``qp`` ensemble that was
written via the DataStore via ``results()``

.. code:: ipython3

    ens = results()

.. code:: ipython3

    expdfids = [2, 180, 13517, 18032]
    fig, axs = plt.subplots(4, 1, figsize=(12,10))
    for i, xx in enumerate(expdfids):
        axs[i].set_xlim(0,3)
        ens[xx].plot_native(axes=axs[i])
    axs[3].set_xlabel("redshift", fontsize=15)




.. parsed-literal::

    Text(0.5, 0, 'redshift')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_16_1.png


GPzEstimator parameterizes each PDF as a single Gaussian, here we see a
few examples of Gaussians of different widths. Now let’s grab the mode
of each PDF (stored as ancil data in the ensemble) and compare to the
true redshifts from the test_data file:

.. code:: ipython3

    truez = test_data.data['photometry']['redshift']
    zmode = ens.ancil['zmode'].flatten()

.. code:: ipython3

    plt.figure(figsize=(12,12))
    plt.scatter(truez, zmode, s=3)
    plt.plot([0,3],[0,3], 'k--')
    plt.xlabel("redshift", fontsize=12)
    plt.ylabel("z mode", fontsize=12)




.. parsed-literal::

    Text(0, 0.5, 'z mode')




.. image:: ../../../docs/rendered/estimation_examples/06_GPz_files/../../../docs/rendered/estimation_examples/06_GPz_19_1.png

