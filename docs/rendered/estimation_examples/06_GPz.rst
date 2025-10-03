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
       1	-3.4303449e-01	 3.2047185e-01	-3.3334839e-01	 3.2091190e-01	[-3.3478445e-01]	 4.6340895e-01


.. parsed-literal::

       2	-2.7054945e-01	 3.0898406e-01	-2.4583189e-01	 3.1025846e-01	[-2.5037873e-01]	 2.3038697e-01


.. parsed-literal::

       3	-2.2767549e-01	 2.8949497e-01	-1.8592686e-01	 2.9161889e-01	[-1.9578146e-01]	 2.7799463e-01
       4	-1.9586498e-01	 2.6555618e-01	-1.5512027e-01	 2.6925362e-01	[-1.7280616e-01]	 1.9717169e-01


.. parsed-literal::

       5	-1.0609800e-01	 2.5708061e-01	-6.9439158e-02	 2.5860611e-01	[-7.6712952e-02]	 2.1246147e-01


.. parsed-literal::

       6	-6.9667108e-02	 2.5112364e-01	-3.7514751e-02	 2.5501359e-01	[-4.9620975e-02]	 2.0756388e-01
       7	-5.0519961e-02	 2.4823186e-01	-2.5899819e-02	 2.5071830e-01	[-3.5386441e-02]	 2.0481205e-01


.. parsed-literal::

       8	-3.8162722e-02	 2.4617489e-01	-1.7317052e-02	 2.4869704e-01	[-2.7558220e-02]	 1.7098498e-01
       9	-2.3230355e-02	 2.4344506e-01	-5.6293474e-03	 2.4618931e-01	[-1.7116834e-02]	 1.7094898e-01


.. parsed-literal::

      10	-1.4409003e-02	 2.4210929e-01	 8.7164431e-04	 2.4531963e-01	[-1.1728446e-02]	 2.1445918e-01
      11	-7.5122917e-03	 2.4069677e-01	 6.8888038e-03	 2.4394921e-01	[-5.6616904e-03]	 1.8183851e-01


.. parsed-literal::

      12	-5.1885939e-03	 2.4025075e-01	 8.9656312e-03	 2.4362841e-01	[-3.7271011e-03]	 2.0121217e-01
      13	 4.0024900e-04	 2.3914704e-01	 1.4553110e-02	 2.4298541e-01	[ 4.3968596e-04]	 2.0070481e-01


.. parsed-literal::

      14	 1.1598348e-02	 2.3621332e-01	 2.7874678e-02	 2.4056709e-01	[ 1.3491060e-02]	 2.0894051e-01


.. parsed-literal::

      15	 3.4452060e-02	 2.2881417e-01	 5.0412166e-02	 2.3392198e-01	[ 3.7391967e-02]	 3.1059551e-01


.. parsed-literal::

      16	 6.1620047e-02	 2.2388285e-01	 7.8536706e-02	 2.2906688e-01	[ 6.7898808e-02]	 2.1093845e-01


.. parsed-literal::

      17	 1.3752304e-01	 2.2206530e-01	 1.5811132e-01	 2.2541099e-01	[ 1.4769279e-01]	 2.0555925e-01
      18	 2.3542522e-01	 2.1809433e-01	 2.6231694e-01	 2.2218440e-01	[ 2.4253874e-01]	 1.9883037e-01


.. parsed-literal::

      19	 2.7627767e-01	 2.1725388e-01	 3.0658489e-01	 2.2185833e-01	[ 2.8897848e-01]	 1.7334986e-01


.. parsed-literal::

      20	 3.1723803e-01	 2.1586221e-01	 3.4859348e-01	 2.2181432e-01	[ 3.3304071e-01]	 2.1755075e-01
      21	 3.5386608e-01	 2.1342971e-01	 3.8498971e-01	 2.1652089e-01	[ 3.7123553e-01]	 1.9443607e-01


.. parsed-literal::

      22	 4.1702259e-01	 2.0930775e-01	 4.4894405e-01	 2.1153658e-01	[ 4.3375438e-01]	 1.9651890e-01
      23	 5.2629557e-01	 2.0708988e-01	 5.6035647e-01	 2.0920890e-01	[ 5.3572648e-01]	 1.9644809e-01


.. parsed-literal::

      24	 5.3068730e-01	 2.1207358e-01	 5.7305351e-01	 2.1509276e-01	  5.1785866e-01 	 2.0979166e-01


.. parsed-literal::

      25	 6.1056704e-01	 2.0724120e-01	 6.4904127e-01	 2.1058571e-01	[ 5.9579406e-01]	 2.1656823e-01


.. parsed-literal::

      26	 6.3655131e-01	 2.0501780e-01	 6.7435199e-01	 2.0725898e-01	[ 6.2394783e-01]	 2.0385194e-01
      27	 6.7238945e-01	 2.0478864e-01	 7.0856522e-01	 2.0621113e-01	[ 6.7093127e-01]	 2.0206547e-01


.. parsed-literal::

      28	 6.9629762e-01	 2.0507302e-01	 7.3291214e-01	 2.0971919e-01	[ 6.8640341e-01]	 2.0218229e-01


.. parsed-literal::

      29	 7.3553085e-01	 2.0123985e-01	 7.7311203e-01	 2.0343524e-01	[ 7.2765460e-01]	 2.1192312e-01


.. parsed-literal::

      30	 7.6139179e-01	 2.0161084e-01	 7.9978507e-01	 2.0436795e-01	[ 7.5312489e-01]	 2.0918012e-01
      31	 7.8518650e-01	 2.0435103e-01	 8.2473308e-01	 2.0909324e-01	[ 7.7208402e-01]	 1.9913101e-01


.. parsed-literal::

      32	 8.0713554e-01	 2.0524825e-01	 8.4772258e-01	 2.1100906e-01	[ 7.8938539e-01]	 2.1085501e-01
      33	 8.2668066e-01	 2.0459430e-01	 8.6888518e-01	 2.1143399e-01	[ 8.0235681e-01]	 1.9961762e-01


.. parsed-literal::

      34	 8.4294755e-01	 2.0320288e-01	 8.8531535e-01	 2.0965990e-01	[ 8.1474369e-01]	 2.1002245e-01


.. parsed-literal::

      35	 8.5718019e-01	 2.0280287e-01	 8.9945060e-01	 2.0948201e-01	[ 8.2918706e-01]	 2.0751357e-01


.. parsed-literal::

      36	 8.7826086e-01	 2.0419424e-01	 9.2102863e-01	 2.1168194e-01	[ 8.5242312e-01]	 2.0819163e-01
      37	 8.9556449e-01	 2.0763644e-01	 9.3836162e-01	 2.1642015e-01	[ 8.7720410e-01]	 1.9180989e-01


.. parsed-literal::

      38	 9.0862578e-01	 2.0661338e-01	 9.5153016e-01	 2.1394469e-01	[ 8.9222013e-01]	 2.0856500e-01


.. parsed-literal::

      39	 9.2385247e-01	 2.0514600e-01	 9.6694052e-01	 2.1113928e-01	[ 9.0441962e-01]	 2.1277237e-01
      40	 9.3728273e-01	 2.0356118e-01	 9.8064486e-01	 2.0887234e-01	[ 9.1770000e-01]	 1.7370296e-01


.. parsed-literal::

      41	 9.5417373e-01	 2.0267064e-01	 9.9869427e-01	 2.0856624e-01	[ 9.3447493e-01]	 2.0461559e-01


.. parsed-literal::

      42	 9.6727474e-01	 2.0185634e-01	 1.0120223e+00	 2.0862060e-01	[ 9.4891903e-01]	 2.0407844e-01
      43	 9.7406925e-01	 2.0105844e-01	 1.0189026e+00	 2.0791190e-01	[ 9.5405220e-01]	 1.8550014e-01


.. parsed-literal::

      44	 9.8981329e-01	 1.9970312e-01	 1.0355749e+00	 2.0738126e-01	[ 9.6246468e-01]	 1.8023276e-01
      45	 1.0025142e+00	 1.9787259e-01	 1.0489031e+00	 2.0700764e-01	[ 9.7135340e-01]	 1.7878127e-01


.. parsed-literal::

      46	 1.0153646e+00	 1.9557090e-01	 1.0621504e+00	 2.0554346e-01	[ 9.8538942e-01]	 1.9861555e-01


.. parsed-literal::

      47	 1.0268690e+00	 1.9432758e-01	 1.0742366e+00	 2.0468782e-01	[ 9.9290720e-01]	 2.0820999e-01


.. parsed-literal::

      48	 1.0390005e+00	 1.9332367e-01	 1.0869337e+00	 2.0379080e-01	[ 1.0098953e+00]	 2.0719790e-01
      49	 1.0516293e+00	 1.9328248e-01	 1.0996264e+00	 2.0300828e-01	[ 1.0189949e+00]	 1.9636536e-01


.. parsed-literal::

      50	 1.0605781e+00	 1.9290619e-01	 1.1085816e+00	 2.0192375e-01	[ 1.0323385e+00]	 2.0079112e-01
      51	 1.0716370e+00	 1.9235738e-01	 1.1195216e+00	 2.0094500e-01	[ 1.0425374e+00]	 2.0314097e-01


.. parsed-literal::

      52	 1.0828744e+00	 1.9025614e-01	 1.1308930e+00	 1.9795764e-01	[ 1.0615020e+00]	 2.0759654e-01


.. parsed-literal::

      53	 1.0933383e+00	 1.8913678e-01	 1.1413044e+00	 1.9672261e-01	[ 1.0709048e+00]	 2.0914841e-01
      54	 1.1052240e+00	 1.8756071e-01	 1.1532429e+00	 1.9476528e-01	[ 1.0807585e+00]	 2.0655918e-01


.. parsed-literal::

      55	 1.1176616e+00	 1.8496344e-01	 1.1662134e+00	 1.9174538e-01	[ 1.0835650e+00]	 2.1524906e-01


.. parsed-literal::

      56	 1.1306804e+00	 1.8482385e-01	 1.1793442e+00	 1.9168896e-01	[ 1.0942571e+00]	 2.0866370e-01


.. parsed-literal::

      57	 1.1408996e+00	 1.8492088e-01	 1.1900753e+00	 1.9168621e-01	[ 1.1037865e+00]	 2.0887899e-01
      58	 1.1584712e+00	 1.8369611e-01	 1.2087244e+00	 1.9008751e-01	[ 1.1195699e+00]	 1.8290234e-01


.. parsed-literal::

      59	 1.1676939e+00	 1.8132334e-01	 1.2193665e+00	 1.8712320e-01	  1.1151425e+00 	 2.0777750e-01
      60	 1.1791140e+00	 1.8019782e-01	 1.2304353e+00	 1.8604588e-01	[ 1.1250328e+00]	 2.0426679e-01


.. parsed-literal::

      61	 1.1901990e+00	 1.7911076e-01	 1.2412890e+00	 1.8539270e-01	[ 1.1300692e+00]	 2.0004153e-01
      62	 1.1982140e+00	 1.7765616e-01	 1.2490294e+00	 1.8412024e-01	[ 1.1387133e+00]	 1.8535209e-01


.. parsed-literal::

      63	 1.2060624e+00	 1.7767394e-01	 1.2572073e+00	 1.8491878e-01	[ 1.1399396e+00]	 2.0434809e-01


.. parsed-literal::

      64	 1.2133849e+00	 1.7586619e-01	 1.2648962e+00	 1.8295887e-01	[ 1.1447840e+00]	 2.0616722e-01


.. parsed-literal::

      65	 1.2202397e+00	 1.7456005e-01	 1.2720361e+00	 1.8129510e-01	[ 1.1535816e+00]	 2.0228219e-01


.. parsed-literal::

      66	 1.2281971e+00	 1.7297571e-01	 1.2804029e+00	 1.7925732e-01	[ 1.1595378e+00]	 2.1115255e-01


.. parsed-literal::

      67	 1.2377877e+00	 1.7194251e-01	 1.2898606e+00	 1.7802923e-01	[ 1.1694932e+00]	 2.0968390e-01


.. parsed-literal::

      68	 1.2451221e+00	 1.7179243e-01	 1.2969871e+00	 1.7821308e-01	[ 1.1747588e+00]	 2.1034336e-01
      69	 1.2531465e+00	 1.7011386e-01	 1.3052094e+00	 1.7653732e-01	[ 1.1766548e+00]	 1.9881320e-01


.. parsed-literal::

      70	 1.2636291e+00	 1.6798688e-01	 1.3154953e+00	 1.7487983e-01	[ 1.1886914e+00]	 2.0124555e-01


.. parsed-literal::

      71	 1.2717763e+00	 1.6624696e-01	 1.3240271e+00	 1.7346034e-01	  1.1861619e+00 	 2.0938683e-01


.. parsed-literal::

      72	 1.2804689e+00	 1.6437803e-01	 1.3328610e+00	 1.7175471e-01	[ 1.1934931e+00]	 2.0550418e-01


.. parsed-literal::

      73	 1.2881365e+00	 1.6279298e-01	 1.3406285e+00	 1.7003162e-01	[ 1.1988130e+00]	 2.1195817e-01
      74	 1.2967914e+00	 1.6188663e-01	 1.3493476e+00	 1.6908762e-01	[ 1.2026461e+00]	 1.9701648e-01


.. parsed-literal::

      75	 1.3061005e+00	 1.6065327e-01	 1.3588216e+00	 1.6737349e-01	[ 1.2124268e+00]	 2.0662332e-01


.. parsed-literal::

      76	 1.3135842e+00	 1.6039361e-01	 1.3662619e+00	 1.6743244e-01	[ 1.2164694e+00]	 2.0736742e-01


.. parsed-literal::

      77	 1.3204542e+00	 1.5945560e-01	 1.3732107e+00	 1.6671213e-01	[ 1.2186130e+00]	 2.0474267e-01


.. parsed-literal::

      78	 1.3302120e+00	 1.5698844e-01	 1.3833336e+00	 1.6455546e-01	  1.2169633e+00 	 2.1376491e-01


.. parsed-literal::

      79	 1.3345854e+00	 1.5524949e-01	 1.3875952e+00	 1.6347035e-01	[ 1.2236020e+00]	 2.0694923e-01


.. parsed-literal::

      80	 1.3423959e+00	 1.5398139e-01	 1.3951818e+00	 1.6244150e-01	[ 1.2264662e+00]	 2.0905852e-01


.. parsed-literal::

      81	 1.3483452e+00	 1.5261556e-01	 1.4009793e+00	 1.6115639e-01	[ 1.2288723e+00]	 2.1674752e-01


.. parsed-literal::

      82	 1.3544526e+00	 1.5074736e-01	 1.4072388e+00	 1.5934507e-01	  1.2256531e+00 	 2.1161342e-01


.. parsed-literal::

      83	 1.3628264e+00	 1.4834829e-01	 1.4156870e+00	 1.5693945e-01	[ 1.2302001e+00]	 2.1505213e-01
      84	 1.3705446e+00	 1.4482826e-01	 1.4236682e+00	 1.5366250e-01	  1.2265697e+00 	 1.9590306e-01


.. parsed-literal::

      85	 1.3775273e+00	 1.4455349e-01	 1.4307503e+00	 1.5341526e-01	[ 1.2313083e+00]	 2.0879197e-01
      86	 1.3864778e+00	 1.4240559e-01	 1.4397801e+00	 1.5146270e-01	[ 1.2407556e+00]	 1.9582891e-01


.. parsed-literal::

      87	 1.3911121e+00	 1.4111740e-01	 1.4446162e+00	 1.5025147e-01	  1.2388651e+00 	 2.1581459e-01


.. parsed-literal::

      88	 1.3960424e+00	 1.4083882e-01	 1.4493906e+00	 1.5011203e-01	[ 1.2477260e+00]	 2.1406984e-01


.. parsed-literal::

      89	 1.4010731e+00	 1.3966402e-01	 1.4545907e+00	 1.4917729e-01	[ 1.2495785e+00]	 2.0226169e-01


.. parsed-literal::

      90	 1.4058749e+00	 1.3922308e-01	 1.4595708e+00	 1.4886597e-01	[ 1.2519377e+00]	 2.1410751e-01


.. parsed-literal::

      91	 1.4127138e+00	 1.3754001e-01	 1.4668194e+00	 1.4775487e-01	[ 1.2570519e+00]	 2.0997906e-01


.. parsed-literal::

      92	 1.4180062e+00	 1.3707160e-01	 1.4722096e+00	 1.4729840e-01	  1.2529549e+00 	 2.0943999e-01


.. parsed-literal::

      93	 1.4217058e+00	 1.3738959e-01	 1.4757606e+00	 1.4751989e-01	[ 1.2607785e+00]	 2.1687198e-01
      94	 1.4255726e+00	 1.3707954e-01	 1.4796692e+00	 1.4705463e-01	[ 1.2665361e+00]	 1.8605375e-01


.. parsed-literal::

      95	 1.4291936e+00	 1.3699016e-01	 1.4832722e+00	 1.4689828e-01	[ 1.2681122e+00]	 2.0397830e-01


.. parsed-literal::

      96	 1.4342635e+00	 1.3691839e-01	 1.4883112e+00	 1.4673143e-01	[ 1.2736736e+00]	 2.2037148e-01
      97	 1.4389123e+00	 1.3668240e-01	 1.4929877e+00	 1.4643319e-01	[ 1.2755885e+00]	 1.8874955e-01


.. parsed-literal::

      98	 1.4431449e+00	 1.3662752e-01	 1.4971251e+00	 1.4621663e-01	[ 1.2812747e+00]	 2.0433712e-01
      99	 1.4466885e+00	 1.3638941e-01	 1.5006416e+00	 1.4578510e-01	[ 1.2855792e+00]	 1.8193817e-01


.. parsed-literal::

     100	 1.4507118e+00	 1.3606564e-01	 1.5048045e+00	 1.4514840e-01	[ 1.2894576e+00]	 2.0553923e-01


.. parsed-literal::

     101	 1.4543415e+00	 1.3577392e-01	 1.5085198e+00	 1.4449504e-01	[ 1.2900860e+00]	 2.1115780e-01


.. parsed-literal::

     102	 1.4573480e+00	 1.3563073e-01	 1.5114981e+00	 1.4438259e-01	[ 1.2924554e+00]	 2.1222806e-01


.. parsed-literal::

     103	 1.4610504e+00	 1.3547505e-01	 1.5153962e+00	 1.4417858e-01	  1.2901227e+00 	 2.0784020e-01
     104	 1.4641243e+00	 1.3545325e-01	 1.5186053e+00	 1.4392379e-01	  1.2882982e+00 	 1.7468047e-01


.. parsed-literal::

     105	 1.4679961e+00	 1.3538712e-01	 1.5226324e+00	 1.4347803e-01	  1.2890717e+00 	 1.9607520e-01


.. parsed-literal::

     106	 1.4711860e+00	 1.3531642e-01	 1.5259250e+00	 1.4324525e-01	  1.2882418e+00 	 2.1140838e-01


.. parsed-literal::

     107	 1.4742029e+00	 1.3510502e-01	 1.5289714e+00	 1.4281640e-01	  1.2894485e+00 	 2.1584105e-01
     108	 1.4763930e+00	 1.3468728e-01	 1.5312862e+00	 1.4224220e-01	  1.2878154e+00 	 1.8521404e-01


.. parsed-literal::

     109	 1.4791473e+00	 1.3457984e-01	 1.5339455e+00	 1.4212582e-01	  1.2903799e+00 	 1.9047570e-01


.. parsed-literal::

     110	 1.4817663e+00	 1.3443524e-01	 1.5365674e+00	 1.4199099e-01	  1.2903629e+00 	 2.0774341e-01
     111	 1.4843918e+00	 1.3430588e-01	 1.5391812e+00	 1.4195018e-01	  1.2907687e+00 	 1.7965221e-01


.. parsed-literal::

     112	 1.4883295e+00	 1.3404450e-01	 1.5433059e+00	 1.4181287e-01	  1.2900913e+00 	 1.8167448e-01


.. parsed-literal::

     113	 1.4913583e+00	 1.3405032e-01	 1.5463243e+00	 1.4187071e-01	  1.2922399e+00 	 2.1718264e-01


.. parsed-literal::

     114	 1.4937616e+00	 1.3391616e-01	 1.5486275e+00	 1.4173368e-01	[ 1.2983340e+00]	 2.0535207e-01


.. parsed-literal::

     115	 1.4961175e+00	 1.3373263e-01	 1.5510387e+00	 1.4142516e-01	[ 1.2995413e+00]	 2.0986366e-01
     116	 1.4979373e+00	 1.3374773e-01	 1.5529530e+00	 1.4131251e-01	[ 1.3019457e+00]	 1.7635322e-01


.. parsed-literal::

     117	 1.5000933e+00	 1.3370438e-01	 1.5551256e+00	 1.4123782e-01	  1.3004571e+00 	 2.1380472e-01


.. parsed-literal::

     118	 1.5028069e+00	 1.3372063e-01	 1.5579045e+00	 1.4126557e-01	  1.2955234e+00 	 2.0982981e-01


.. parsed-literal::

     119	 1.5046606e+00	 1.3370564e-01	 1.5598187e+00	 1.4129573e-01	  1.2944668e+00 	 2.1240163e-01
     120	 1.5073934e+00	 1.3367864e-01	 1.5626651e+00	 1.4136407e-01	  1.2942098e+00 	 2.0819402e-01


.. parsed-literal::

     121	 1.5094605e+00	 1.3357105e-01	 1.5648368e+00	 1.4124483e-01	  1.2938469e+00 	 2.0870852e-01


.. parsed-literal::

     122	 1.5118543e+00	 1.3350387e-01	 1.5671942e+00	 1.4118284e-01	  1.2998774e+00 	 2.1514320e-01


.. parsed-literal::

     123	 1.5135802e+00	 1.3347104e-01	 1.5689497e+00	 1.4106891e-01	  1.3008245e+00 	 2.1077108e-01


.. parsed-literal::

     124	 1.5151472e+00	 1.3345189e-01	 1.5705762e+00	 1.4100663e-01	[ 1.3028994e+00]	 2.1367335e-01


.. parsed-literal::

     125	 1.5167180e+00	 1.3344118e-01	 1.5721638e+00	 1.4104656e-01	  1.3009592e+00 	 2.2040391e-01


.. parsed-literal::

     126	 1.5188478e+00	 1.3337147e-01	 1.5743527e+00	 1.4107860e-01	  1.2986551e+00 	 2.0411038e-01


.. parsed-literal::

     127	 1.5208212e+00	 1.3321521e-01	 1.5764670e+00	 1.4120419e-01	  1.2912310e+00 	 2.1011233e-01
     128	 1.5228221e+00	 1.3313011e-01	 1.5785056e+00	 1.4114440e-01	  1.2937036e+00 	 1.9963717e-01


.. parsed-literal::

     129	 1.5242864e+00	 1.3300127e-01	 1.5799638e+00	 1.4096910e-01	  1.2935926e+00 	 2.1054268e-01
     130	 1.5260065e+00	 1.3288131e-01	 1.5817131e+00	 1.4081647e-01	  1.2922664e+00 	 1.7284727e-01


.. parsed-literal::

     131	 1.5277009e+00	 1.3277173e-01	 1.5835268e+00	 1.4060902e-01	  1.2869065e+00 	 1.7970562e-01


.. parsed-literal::

     132	 1.5292120e+00	 1.3279828e-01	 1.5850465e+00	 1.4066722e-01	  1.2873275e+00 	 2.1350765e-01


.. parsed-literal::

     133	 1.5305330e+00	 1.3286760e-01	 1.5864013e+00	 1.4079351e-01	  1.2876500e+00 	 2.1791720e-01


.. parsed-literal::

     134	 1.5318143e+00	 1.3292892e-01	 1.5877159e+00	 1.4090383e-01	  1.2884488e+00 	 2.0465851e-01
     135	 1.5340039e+00	 1.3308201e-01	 1.5900095e+00	 1.4121188e-01	  1.2899025e+00 	 2.0133042e-01


.. parsed-literal::

     136	 1.5359260e+00	 1.3305801e-01	 1.5918932e+00	 1.4126328e-01	  1.2939445e+00 	 2.0995975e-01


.. parsed-literal::

     137	 1.5375590e+00	 1.3301177e-01	 1.5935068e+00	 1.4125960e-01	  1.2954139e+00 	 2.0982265e-01
     138	 1.5394089e+00	 1.3291448e-01	 1.5953558e+00	 1.4120567e-01	  1.2971426e+00 	 1.9734144e-01


.. parsed-literal::

     139	 1.5404666e+00	 1.3288846e-01	 1.5965112e+00	 1.4119977e-01	  1.2959999e+00 	 2.0451570e-01


.. parsed-literal::

     140	 1.5422170e+00	 1.3286570e-01	 1.5981887e+00	 1.4120928e-01	  1.2966152e+00 	 2.0320654e-01
     141	 1.5432218e+00	 1.3284759e-01	 1.5992173e+00	 1.4121618e-01	  1.2962812e+00 	 1.9888687e-01


.. parsed-literal::

     142	 1.5441544e+00	 1.3279868e-01	 1.6001925e+00	 1.4121590e-01	  1.2939227e+00 	 2.0846939e-01
     143	 1.5451257e+00	 1.3268641e-01	 1.6012998e+00	 1.4122272e-01	  1.2902811e+00 	 1.9519758e-01


.. parsed-literal::

     144	 1.5465279e+00	 1.3264054e-01	 1.6026176e+00	 1.4122086e-01	  1.2885942e+00 	 1.7961049e-01


.. parsed-literal::

     145	 1.5472258e+00	 1.3259367e-01	 1.6032869e+00	 1.4121309e-01	  1.2881242e+00 	 2.0277691e-01


.. parsed-literal::

     146	 1.5478730e+00	 1.3255081e-01	 1.6039364e+00	 1.4121043e-01	  1.2876859e+00 	 2.0558095e-01
     147	 1.5495886e+00	 1.3249231e-01	 1.6056945e+00	 1.4119373e-01	  1.2864375e+00 	 1.8939567e-01


.. parsed-literal::

     148	 1.5503911e+00	 1.3235717e-01	 1.6067431e+00	 1.4093243e-01	  1.2832559e+00 	 2.0434427e-01


.. parsed-literal::

     149	 1.5528492e+00	 1.3240491e-01	 1.6091431e+00	 1.4098516e-01	  1.2817062e+00 	 2.1275234e-01


.. parsed-literal::

     150	 1.5538102e+00	 1.3243856e-01	 1.6101123e+00	 1.4093192e-01	  1.2820600e+00 	 2.0540762e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 2s, sys: 1.11 s, total: 2min 4s
    Wall time: 31.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fd7005be5f0>



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
    CPU times: user 1.8 s, sys: 51.9 ms, total: 1.86 s
    Wall time: 585 ms


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

