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
       1	-3.4451244e-01	 3.2096299e-01	-3.3487055e-01	 3.1924087e-01	[-3.3158750e-01]	 4.6145391e-01


.. parsed-literal::

       2	-2.7514042e-01	 3.1092047e-01	-2.5172472e-01	 3.0846730e-01	[-2.4443395e-01]	 2.3113155e-01


.. parsed-literal::

       3	-2.3137985e-01	 2.9015310e-01	-1.8949283e-01	 2.8709214e-01	[-1.7456701e-01]	 2.8145671e-01
       4	-1.9062743e-01	 2.6695570e-01	-1.4891057e-01	 2.6170880e-01	[-1.1624864e-01]	 1.7813683e-01


.. parsed-literal::

       5	-1.0604351e-01	 2.5772653e-01	-7.4276458e-02	 2.5267885e-01	[-5.0186040e-02]	 1.9985080e-01


.. parsed-literal::

       6	-7.1808712e-02	 2.5235670e-01	-4.3528777e-02	 2.4998226e-01	[-3.1696006e-02]	 2.0924139e-01


.. parsed-literal::

       7	-5.6253773e-02	 2.4990547e-01	-3.2717287e-02	 2.4706506e-01	[-1.9560621e-02]	 2.0999265e-01
       8	-4.2643232e-02	 2.4759271e-01	-2.2984456e-02	 2.4425215e-01	[-7.9148888e-03]	 1.9022322e-01


.. parsed-literal::

       9	-3.0734577e-02	 2.4537449e-01	-1.3477068e-02	 2.4136742e-01	[ 4.0442511e-03]	 2.1028018e-01


.. parsed-literal::

      10	-2.2002317e-02	 2.4375207e-01	-6.9217751e-03	 2.3851806e-01	[ 1.6629289e-02]	 2.1254110e-01


.. parsed-literal::

      11	-1.4629922e-02	 2.4239440e-01	-3.3021594e-04	 2.3750867e-01	[ 2.0509159e-02]	 2.1371102e-01


.. parsed-literal::

      12	-1.2390595e-02	 2.4204198e-01	 1.6264812e-03	 2.3727694e-01	[ 2.2517542e-02]	 2.1295428e-01


.. parsed-literal::

      13	-6.8749663e-03	 2.4105040e-01	 7.0456293e-03	 2.3662732e-01	[ 2.6372732e-02]	 2.2194195e-01


.. parsed-literal::

      14	 9.2639561e-02	 2.2878805e-01	 1.1195608e-01	 2.2486531e-01	[ 1.2902424e-01]	 3.1095982e-01


.. parsed-literal::

      15	 1.6354724e-01	 2.2376682e-01	 1.8721318e-01	 2.2069785e-01	[ 2.0291056e-01]	 3.2280588e-01


.. parsed-literal::

      16	 2.2138530e-01	 2.2030265e-01	 2.4792057e-01	 2.1646177e-01	[ 2.6729020e-01]	 2.0388889e-01


.. parsed-literal::

      17	 2.9293570e-01	 2.1857071e-01	 3.2327762e-01	 2.1478716e-01	[ 3.5321749e-01]	 2.0386887e-01
      18	 3.4586006e-01	 2.1275783e-01	 3.7956646e-01	 2.0646671e-01	[ 4.2646768e-01]	 2.0006967e-01


.. parsed-literal::

      19	 4.0225772e-01	 2.0985631e-01	 4.3627358e-01	 2.0332113e-01	[ 4.8482506e-01]	 2.0647049e-01


.. parsed-literal::

      20	 4.6040600e-01	 2.0625606e-01	 4.9470413e-01	 1.9870715e-01	[ 5.4262149e-01]	 2.1287298e-01


.. parsed-literal::

      21	 5.7210699e-01	 2.0172111e-01	 6.0813347e-01	 1.9558700e-01	[ 6.3970283e-01]	 2.1500349e-01


.. parsed-literal::

      22	 6.2638993e-01	 2.0072474e-01	 6.6595827e-01	 1.9250072e-01	[ 6.5584676e-01]	 2.0721364e-01
      23	 6.6218753e-01	 1.9823421e-01	 7.0042750e-01	 1.9089347e-01	[ 6.7206877e-01]	 1.8202257e-01


.. parsed-literal::

      24	 6.9896064e-01	 1.9487329e-01	 7.3735577e-01	 1.8735393e-01	[ 7.1655711e-01]	 2.0716906e-01


.. parsed-literal::

      25	 7.2993588e-01	 1.9500510e-01	 7.6724673e-01	 1.8895183e-01	[ 7.6072994e-01]	 2.1203017e-01
      26	 7.6230325e-01	 1.9464575e-01	 7.9922482e-01	 1.8939200e-01	[ 7.8791896e-01]	 1.7110205e-01


.. parsed-literal::

      27	 7.8896732e-01	 1.8997417e-01	 8.2718709e-01	 1.8504898e-01	[ 8.0746766e-01]	 1.8515539e-01
      28	 8.1547757e-01	 1.8700231e-01	 8.5524128e-01	 1.8290400e-01	[ 8.3368904e-01]	 1.6823149e-01


.. parsed-literal::

      29	 8.4234053e-01	 1.8276328e-01	 8.8246768e-01	 1.7934579e-01	[ 8.5770128e-01]	 2.1234703e-01
      30	 8.6897746e-01	 1.7982355e-01	 9.0919137e-01	 1.7733522e-01	[ 8.9070634e-01]	 1.9428277e-01


.. parsed-literal::

      31	 9.0156778e-01	 1.7706651e-01	 9.4307708e-01	 1.7404597e-01	[ 9.2078143e-01]	 2.1243119e-01


.. parsed-literal::

      32	 9.2445151e-01	 1.7670165e-01	 9.6735147e-01	 1.7359571e-01	[ 9.5409650e-01]	 2.1890807e-01
      33	 9.4631203e-01	 1.7536013e-01	 9.8912900e-01	 1.7134766e-01	[ 9.7708015e-01]	 1.8407106e-01


.. parsed-literal::

      34	 9.5758742e-01	 1.7463443e-01	 1.0005284e+00	 1.7022213e-01	[ 9.8756012e-01]	 2.0800066e-01


.. parsed-literal::

      35	 9.7821296e-01	 1.7401185e-01	 1.0221595e+00	 1.6816266e-01	[ 1.0083504e+00]	 2.1146917e-01


.. parsed-literal::

      36	 9.9316582e-01	 1.7301014e-01	 1.0375929e+00	 1.6706196e-01	[ 1.0235257e+00]	 2.1021390e-01


.. parsed-literal::

      37	 1.0069893e+00	 1.7189575e-01	 1.0514695e+00	 1.6604804e-01	[ 1.0384245e+00]	 2.0835257e-01


.. parsed-literal::

      38	 1.0216304e+00	 1.7019704e-01	 1.0667697e+00	 1.6454735e-01	[ 1.0502583e+00]	 2.1660948e-01


.. parsed-literal::

      39	 1.0358865e+00	 1.6856738e-01	 1.0814359e+00	 1.6325380e-01	[ 1.0581250e+00]	 2.1898627e-01


.. parsed-literal::

      40	 1.0510954e+00	 1.6679321e-01	 1.0977826e+00	 1.6179964e-01	[ 1.0637081e+00]	 2.1626616e-01
      41	 1.0643882e+00	 1.6601438e-01	 1.1112034e+00	 1.6176220e-01	[ 1.0783573e+00]	 1.9269681e-01


.. parsed-literal::

      42	 1.0786659e+00	 1.6424100e-01	 1.1255881e+00	 1.6106332e-01	[ 1.0904339e+00]	 1.9865298e-01
      43	 1.0927609e+00	 1.6107074e-01	 1.1404342e+00	 1.5801008e-01	[ 1.1016717e+00]	 1.9932532e-01


.. parsed-literal::

      44	 1.1055047e+00	 1.5950969e-01	 1.1529525e+00	 1.5698541e-01	[ 1.1099203e+00]	 2.1167088e-01


.. parsed-literal::

      45	 1.1149001e+00	 1.5812429e-01	 1.1619611e+00	 1.5603809e-01	[ 1.1172605e+00]	 2.0467257e-01


.. parsed-literal::

      46	 1.1245860e+00	 1.5569977e-01	 1.1717955e+00	 1.5451082e-01	[ 1.1207589e+00]	 2.0401692e-01


.. parsed-literal::

      47	 1.1334619e+00	 1.5462666e-01	 1.1807686e+00	 1.5381806e-01	[ 1.1248654e+00]	 2.0818520e-01
      48	 1.1399735e+00	 1.5418725e-01	 1.1872729e+00	 1.5345529e-01	[ 1.1297931e+00]	 1.6824365e-01


.. parsed-literal::

      49	 1.1520367e+00	 1.5309459e-01	 1.1996809e+00	 1.5274895e-01	[ 1.1363637e+00]	 2.2089767e-01


.. parsed-literal::

      50	 1.1641353e+00	 1.5230350e-01	 1.2123092e+00	 1.5251789e-01	[ 1.1430106e+00]	 2.1050167e-01
      51	 1.1748236e+00	 1.5128728e-01	 1.2231267e+00	 1.5169891e-01	[ 1.1472407e+00]	 2.0017457e-01


.. parsed-literal::

      52	 1.1836041e+00	 1.5007034e-01	 1.2318846e+00	 1.5045751e-01	[ 1.1550231e+00]	 1.8417859e-01


.. parsed-literal::

      53	 1.1957574e+00	 1.4888753e-01	 1.2445707e+00	 1.4879418e-01	[ 1.1637162e+00]	 2.1094275e-01


.. parsed-literal::

      54	 1.2058543e+00	 1.4790137e-01	 1.2550279e+00	 1.4741824e-01	[ 1.1727330e+00]	 2.0428085e-01
      55	 1.2152729e+00	 1.4738879e-01	 1.2644461e+00	 1.4664469e-01	[ 1.1824751e+00]	 1.7736864e-01


.. parsed-literal::

      56	 1.2254563e+00	 1.4686740e-01	 1.2747072e+00	 1.4618313e-01	[ 1.1905710e+00]	 2.1443272e-01


.. parsed-literal::

      57	 1.2358435e+00	 1.4628309e-01	 1.2852668e+00	 1.4606933e-01	[ 1.1949462e+00]	 2.0720887e-01


.. parsed-literal::

      58	 1.2466239e+00	 1.4552376e-01	 1.2966124e+00	 1.4725538e-01	  1.1856651e+00 	 2.1041489e-01


.. parsed-literal::

      59	 1.2588003e+00	 1.4466977e-01	 1.3084855e+00	 1.4701014e-01	[ 1.1974471e+00]	 2.1049738e-01


.. parsed-literal::

      60	 1.2655327e+00	 1.4427470e-01	 1.3154020e+00	 1.4658353e-01	[ 1.1988944e+00]	 2.0866060e-01
      61	 1.2759048e+00	 1.4375803e-01	 1.3261186e+00	 1.4606678e-01	[ 1.1998509e+00]	 1.9765615e-01


.. parsed-literal::

      62	 1.2837781e+00	 1.4297078e-01	 1.3343593e+00	 1.4516046e-01	  1.1902421e+00 	 2.0125628e-01


.. parsed-literal::

      63	 1.2938739e+00	 1.4239306e-01	 1.3442485e+00	 1.4445273e-01	[ 1.2079046e+00]	 2.1134543e-01


.. parsed-literal::

      64	 1.2992589e+00	 1.4206566e-01	 1.3496709e+00	 1.4388564e-01	[ 1.2156611e+00]	 2.0957732e-01


.. parsed-literal::

      65	 1.3085103e+00	 1.4104393e-01	 1.3594387e+00	 1.4260498e-01	[ 1.2277352e+00]	 2.1042991e-01


.. parsed-literal::

      66	 1.3143565e+00	 1.4192325e-01	 1.3651299e+00	 1.4250653e-01	[ 1.2417937e+00]	 2.1077299e-01


.. parsed-literal::

      67	 1.3205075e+00	 1.4119940e-01	 1.3711579e+00	 1.4212480e-01	[ 1.2470200e+00]	 2.1054697e-01


.. parsed-literal::

      68	 1.3272818e+00	 1.4026366e-01	 1.3780395e+00	 1.4156309e-01	[ 1.2514285e+00]	 2.0417285e-01
      69	 1.3320498e+00	 1.3990614e-01	 1.3828711e+00	 1.4152715e-01	[ 1.2516219e+00]	 1.7295074e-01


.. parsed-literal::

      70	 1.3401481e+00	 1.3931776e-01	 1.3912568e+00	 1.4129008e-01	[ 1.2542270e+00]	 2.0087028e-01


.. parsed-literal::

      71	 1.3456953e+00	 1.3877372e-01	 1.3969024e+00	 1.4103386e-01	[ 1.2604621e+00]	 2.1032500e-01
      72	 1.3512890e+00	 1.3888903e-01	 1.4024379e+00	 1.4114602e-01	[ 1.2670194e+00]	 1.8825507e-01


.. parsed-literal::

      73	 1.3583154e+00	 1.3870479e-01	 1.4095984e+00	 1.4107780e-01	[ 1.2733234e+00]	 2.0944834e-01
      74	 1.3639507e+00	 1.3883098e-01	 1.4153830e+00	 1.4142101e-01	[ 1.2783311e+00]	 1.9832683e-01


.. parsed-literal::

      75	 1.3672191e+00	 1.3941890e-01	 1.4186387e+00	 1.4162912e-01	[ 1.2861132e+00]	 2.1308756e-01


.. parsed-literal::

      76	 1.3733466e+00	 1.3896899e-01	 1.4246685e+00	 1.4151736e-01	[ 1.2926876e+00]	 2.1003604e-01
      77	 1.3765950e+00	 1.3864481e-01	 1.4279561e+00	 1.4127710e-01	[ 1.2962735e+00]	 1.8790030e-01


.. parsed-literal::

      78	 1.3818588e+00	 1.3805046e-01	 1.4334271e+00	 1.4086721e-01	[ 1.2999617e+00]	 2.0865035e-01
      79	 1.3874810e+00	 1.3742472e-01	 1.4392537e+00	 1.4016546e-01	[ 1.3053002e+00]	 2.0007277e-01


.. parsed-literal::

      80	 1.3932179e+00	 1.3689592e-01	 1.4450355e+00	 1.4008764e-01	[ 1.3090339e+00]	 1.8755651e-01
      81	 1.3964472e+00	 1.3681860e-01	 1.4481908e+00	 1.4016537e-01	[ 1.3111249e+00]	 1.7681599e-01


.. parsed-literal::

      82	 1.4003895e+00	 1.3646881e-01	 1.4521605e+00	 1.4016538e-01	  1.3108509e+00 	 2.0634246e-01


.. parsed-literal::

      83	 1.4039536e+00	 1.3624488e-01	 1.4559090e+00	 1.4020195e-01	  1.3069854e+00 	 2.0771694e-01
      84	 1.4088430e+00	 1.3593708e-01	 1.4607481e+00	 1.4014512e-01	  1.3080089e+00 	 1.9831896e-01


.. parsed-literal::

      85	 1.4120546e+00	 1.3567554e-01	 1.4640177e+00	 1.4000338e-01	  1.3088082e+00 	 1.7733216e-01


.. parsed-literal::

      86	 1.4151622e+00	 1.3561803e-01	 1.4672686e+00	 1.3986553e-01	  1.3061055e+00 	 2.2003579e-01


.. parsed-literal::

      87	 1.4180429e+00	 1.3541526e-01	 1.4703150e+00	 1.3946482e-01	  1.3028851e+00 	 2.1118331e-01


.. parsed-literal::

      88	 1.4214744e+00	 1.3543019e-01	 1.4737334e+00	 1.3936090e-01	  1.3034154e+00 	 2.1140575e-01


.. parsed-literal::

      89	 1.4252036e+00	 1.3560097e-01	 1.4775135e+00	 1.3917502e-01	  1.3013984e+00 	 2.1042490e-01


.. parsed-literal::

      90	 1.4275063e+00	 1.3558755e-01	 1.4798408e+00	 1.3901824e-01	  1.3013010e+00 	 2.0205331e-01


.. parsed-literal::

      91	 1.4302534e+00	 1.3583477e-01	 1.4826832e+00	 1.3868212e-01	  1.3025161e+00 	 3.1976175e-01


.. parsed-literal::

      92	 1.4337689e+00	 1.3549136e-01	 1.4862499e+00	 1.3836555e-01	  1.3037790e+00 	 2.1638894e-01


.. parsed-literal::

      93	 1.4357746e+00	 1.3534173e-01	 1.4882493e+00	 1.3820243e-01	  1.3074730e+00 	 2.1504021e-01
      94	 1.4393942e+00	 1.3494419e-01	 1.4919324e+00	 1.3768902e-01	[ 1.3137166e+00]	 1.7532301e-01


.. parsed-literal::

      95	 1.4422000e+00	 1.3492866e-01	 1.4947476e+00	 1.3732809e-01	[ 1.3165082e+00]	 2.0811462e-01


.. parsed-literal::

      96	 1.4449412e+00	 1.3480031e-01	 1.4974312e+00	 1.3713595e-01	[ 1.3193102e+00]	 2.0595622e-01


.. parsed-literal::

      97	 1.4492546e+00	 1.3454191e-01	 1.5017312e+00	 1.3684406e-01	[ 1.3219237e+00]	 2.1202707e-01
      98	 1.4515081e+00	 1.3444331e-01	 1.5040332e+00	 1.3683643e-01	  1.3206303e+00 	 1.8829131e-01


.. parsed-literal::

      99	 1.4541391e+00	 1.3428182e-01	 1.5066528e+00	 1.3686641e-01	  1.3200941e+00 	 2.0127463e-01


.. parsed-literal::

     100	 1.4582001e+00	 1.3428149e-01	 1.5108659e+00	 1.3723943e-01	  1.3151099e+00 	 2.0642686e-01
     101	 1.4610412e+00	 1.3411254e-01	 1.5137301e+00	 1.3719962e-01	  1.3150414e+00 	 1.9854116e-01


.. parsed-literal::

     102	 1.4630216e+00	 1.3412191e-01	 1.5156940e+00	 1.3711957e-01	  1.3175316e+00 	 2.1822810e-01


.. parsed-literal::

     103	 1.4668546e+00	 1.3418654e-01	 1.5196380e+00	 1.3697413e-01	  1.3199343e+00 	 2.1711564e-01


.. parsed-literal::

     104	 1.4691647e+00	 1.3428458e-01	 1.5220134e+00	 1.3688265e-01	  1.3165972e+00 	 2.0625567e-01


.. parsed-literal::

     105	 1.4718553e+00	 1.3428140e-01	 1.5247247e+00	 1.3670559e-01	  1.3150649e+00 	 2.3104763e-01
     106	 1.4755796e+00	 1.3432332e-01	 1.5285270e+00	 1.3635437e-01	  1.3139224e+00 	 1.8873167e-01


.. parsed-literal::

     107	 1.4771570e+00	 1.3444179e-01	 1.5302500e+00	 1.3638027e-01	  1.3068393e+00 	 1.9003510e-01
     108	 1.4799033e+00	 1.3437769e-01	 1.5329296e+00	 1.3603935e-01	  1.3142994e+00 	 1.9835973e-01


.. parsed-literal::

     109	 1.4812532e+00	 1.3437173e-01	 1.5343001e+00	 1.3597873e-01	  1.3177670e+00 	 1.9782567e-01


.. parsed-literal::

     110	 1.4831907e+00	 1.3438143e-01	 1.5362896e+00	 1.3588498e-01	  1.3210700e+00 	 2.0662713e-01


.. parsed-literal::

     111	 1.4855323e+00	 1.3431231e-01	 1.5387267e+00	 1.3558361e-01	[ 1.3250066e+00]	 2.1295476e-01


.. parsed-literal::

     112	 1.4884313e+00	 1.3431594e-01	 1.5416339e+00	 1.3538689e-01	[ 1.3270152e+00]	 2.0273352e-01
     113	 1.4904238e+00	 1.3424581e-01	 1.5436002e+00	 1.3525157e-01	  1.3256986e+00 	 2.0008707e-01


.. parsed-literal::

     114	 1.4925267e+00	 1.3413100e-01	 1.5457458e+00	 1.3508379e-01	  1.3219735e+00 	 2.0361543e-01


.. parsed-literal::

     115	 1.4942256e+00	 1.3397049e-01	 1.5475449e+00	 1.3502704e-01	  1.3177157e+00 	 3.1700087e-01


.. parsed-literal::

     116	 1.4963109e+00	 1.3390253e-01	 1.5497362e+00	 1.3486646e-01	  1.3120906e+00 	 2.1496129e-01


.. parsed-literal::

     117	 1.4978883e+00	 1.3379811e-01	 1.5513678e+00	 1.3472710e-01	  1.3116165e+00 	 2.0887852e-01


.. parsed-literal::

     118	 1.5001307e+00	 1.3367062e-01	 1.5536951e+00	 1.3464682e-01	  1.3105744e+00 	 2.1207714e-01


.. parsed-literal::

     119	 1.5016159e+00	 1.3352337e-01	 1.5552501e+00	 1.3435228e-01	  1.3142859e+00 	 2.0420742e-01


.. parsed-literal::

     120	 1.5031474e+00	 1.3343644e-01	 1.5567666e+00	 1.3445930e-01	  1.3145551e+00 	 2.0642686e-01


.. parsed-literal::

     121	 1.5046048e+00	 1.3345548e-01	 1.5581724e+00	 1.3450974e-01	  1.3158121e+00 	 2.0475602e-01


.. parsed-literal::

     122	 1.5060490e+00	 1.3347534e-01	 1.5596384e+00	 1.3450954e-01	  1.3175729e+00 	 2.0924139e-01


.. parsed-literal::

     123	 1.5080861e+00	 1.3363041e-01	 1.5617344e+00	 1.3436395e-01	  1.3181773e+00 	 2.0684648e-01


.. parsed-literal::

     124	 1.5095937e+00	 1.3369701e-01	 1.5632910e+00	 1.3430345e-01	  1.3186398e+00 	 2.1954155e-01


.. parsed-literal::

     125	 1.5110878e+00	 1.3369101e-01	 1.5648534e+00	 1.3418969e-01	  1.3183376e+00 	 2.1178412e-01


.. parsed-literal::

     126	 1.5122031e+00	 1.3376559e-01	 1.5659812e+00	 1.3423148e-01	  1.3156080e+00 	 2.0981479e-01
     127	 1.5133473e+00	 1.3373159e-01	 1.5671281e+00	 1.3431736e-01	  1.3135214e+00 	 1.9968128e-01


.. parsed-literal::

     128	 1.5158340e+00	 1.3365538e-01	 1.5696384e+00	 1.3463683e-01	  1.3079395e+00 	 1.8787456e-01


.. parsed-literal::

     129	 1.5169957e+00	 1.3360287e-01	 1.5708788e+00	 1.3488996e-01	  1.3070148e+00 	 2.0525765e-01
     130	 1.5182485e+00	 1.3356108e-01	 1.5721123e+00	 1.3489446e-01	  1.3074247e+00 	 1.7003322e-01


.. parsed-literal::

     131	 1.5193340e+00	 1.3350322e-01	 1.5732257e+00	 1.3488243e-01	  1.3082086e+00 	 2.1907640e-01


.. parsed-literal::

     132	 1.5201647e+00	 1.3342735e-01	 1.5740954e+00	 1.3484388e-01	  1.3076217e+00 	 2.0803285e-01
     133	 1.5223895e+00	 1.3320161e-01	 1.5764273e+00	 1.3472293e-01	  1.3048062e+00 	 1.9089484e-01


.. parsed-literal::

     134	 1.5235198e+00	 1.3309890e-01	 1.5776094e+00	 1.3458878e-01	  1.3009738e+00 	 3.2775640e-01
     135	 1.5248578e+00	 1.3296576e-01	 1.5789476e+00	 1.3458064e-01	  1.2990163e+00 	 1.8484926e-01


.. parsed-literal::

     136	 1.5258906e+00	 1.3286081e-01	 1.5799455e+00	 1.3446828e-01	  1.2980942e+00 	 2.1718836e-01
     137	 1.5269862e+00	 1.3272405e-01	 1.5810333e+00	 1.3445027e-01	  1.2981609e+00 	 1.9887710e-01


.. parsed-literal::

     138	 1.5287190e+00	 1.3252521e-01	 1.5827548e+00	 1.3428369e-01	  1.2989929e+00 	 1.9821858e-01


.. parsed-literal::

     139	 1.5299907e+00	 1.3232961e-01	 1.5840797e+00	 1.3410138e-01	  1.2997519e+00 	 2.1062660e-01


.. parsed-literal::

     140	 1.5311485e+00	 1.3225719e-01	 1.5852425e+00	 1.3401590e-01	  1.3010414e+00 	 2.1639657e-01
     141	 1.5327307e+00	 1.3207887e-01	 1.5868693e+00	 1.3381046e-01	  1.3015059e+00 	 1.8634009e-01


.. parsed-literal::

     142	 1.5339710e+00	 1.3189470e-01	 1.5881430e+00	 1.3365898e-01	  1.2997170e+00 	 2.1432042e-01


.. parsed-literal::

     143	 1.5356024e+00	 1.3163611e-01	 1.5898618e+00	 1.3352364e-01	  1.2960183e+00 	 2.0915842e-01


.. parsed-literal::

     144	 1.5369596e+00	 1.3152630e-01	 1.5912086e+00	 1.3351915e-01	  1.2898895e+00 	 2.1129894e-01
     145	 1.5377961e+00	 1.3155248e-01	 1.5920031e+00	 1.3358763e-01	  1.2904130e+00 	 1.7925811e-01


.. parsed-literal::

     146	 1.5388116e+00	 1.3159448e-01	 1.5930345e+00	 1.3397169e-01	  1.2888423e+00 	 1.7138100e-01


.. parsed-literal::

     147	 1.5399263e+00	 1.3152083e-01	 1.5941620e+00	 1.3392696e-01	  1.2885197e+00 	 2.0779991e-01


.. parsed-literal::

     148	 1.5408438e+00	 1.3141563e-01	 1.5951388e+00	 1.3389787e-01	  1.2876745e+00 	 2.1061492e-01


.. parsed-literal::

     149	 1.5420950e+00	 1.3123529e-01	 1.5965111e+00	 1.3388011e-01	  1.2850402e+00 	 2.1067786e-01


.. parsed-literal::

     150	 1.5434814e+00	 1.3109220e-01	 1.5979820e+00	 1.3395204e-01	  1.2835365e+00 	 2.1485686e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 5s, sys: 1.16 s, total: 2min 6s
    Wall time: 31.7 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fa8ffc8fe20>



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
    CPU times: user 1.78 s, sys: 57.9 ms, total: 1.84 s
    Wall time: 579 ms


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

