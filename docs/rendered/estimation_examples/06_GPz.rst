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
       1	-3.4680087e-01	 3.2171397e-01	-3.3705767e-01	 3.1617810e-01	[-3.2578126e-01]	 4.7168779e-01


.. parsed-literal::

       2	-2.7337300e-01	 3.0965923e-01	-2.4788241e-01	 3.0651993e-01	[-2.3427410e-01]	 2.3261738e-01


.. parsed-literal::

       3	-2.2829146e-01	 2.8874301e-01	-1.8430843e-01	 2.8792388e-01	[-1.7737473e-01]	 2.8310728e-01


.. parsed-literal::

       4	-1.8759056e-01	 2.7074545e-01	-1.3756857e-01	 2.7347507e-01	[-1.4801661e-01]	 3.2399297e-01


.. parsed-literal::

       5	-1.3233940e-01	 2.5734367e-01	-1.0159943e-01	 2.6150228e-01	[-1.1194746e-01]	 2.1153307e-01


.. parsed-literal::

       6	-7.2295905e-02	 2.5296021e-01	-4.4214478e-02	 2.5454720e-01	[-4.9666132e-02]	 2.1742845e-01


.. parsed-literal::

       7	-5.4815048e-02	 2.4924494e-01	-3.0707338e-02	 2.5073923e-01	[-3.3420440e-02]	 2.1706891e-01


.. parsed-literal::

       8	-4.3002489e-02	 2.4737223e-01	-2.2248581e-02	 2.4811117e-01	[-2.3214108e-02]	 2.1252275e-01


.. parsed-literal::

       9	-2.6480585e-02	 2.4425510e-01	-9.1997746e-03	 2.4368690e-01	[-5.9419665e-03]	 2.2105122e-01
      10	-1.6217831e-02	 2.4234186e-01	-8.2170758e-04	 2.4216349e-01	[ 7.5062692e-05]	 1.9993591e-01


.. parsed-literal::

      11	-1.1569684e-02	 2.4152157e-01	 3.1101947e-03	 2.4204398e-01	[ 1.2710406e-03]	 2.1862316e-01
      12	-7.2537866e-03	 2.4073471e-01	 6.9366874e-03	 2.4165432e-01	[ 3.6510066e-03]	 1.8972635e-01


.. parsed-literal::

      13	-1.6036584e-03	 2.3956318e-01	 1.2679637e-02	 2.4094296e-01	[ 7.9609819e-03]	 2.0082211e-01


.. parsed-literal::

      14	 8.1263101e-02	 2.2545269e-01	 1.0137223e-01	 2.3254711e-01	[ 9.1378482e-02]	 3.3466220e-01
      15	 1.2200064e-01	 2.2243696e-01	 1.4284292e-01	 2.2592946e-01	[ 1.3832672e-01]	 1.9426990e-01


.. parsed-literal::

      16	 1.9552611e-01	 2.1713076e-01	 2.1907043e-01	 2.2174071e-01	[ 2.1116481e-01]	 2.0556974e-01


.. parsed-literal::

      17	 2.6518866e-01	 2.1526898e-01	 2.9538532e-01	 2.2102281e-01	[ 2.8765414e-01]	 2.1302748e-01


.. parsed-literal::

      18	 3.2072263e-01	 2.1312702e-01	 3.5195964e-01	 2.2065567e-01	[ 3.4780159e-01]	 2.2339177e-01


.. parsed-literal::

      19	 3.7044730e-01	 2.1081965e-01	 4.0176918e-01	 2.1618791e-01	[ 4.0651746e-01]	 2.1831608e-01


.. parsed-literal::

      20	 4.3534723e-01	 2.0647146e-01	 4.6802643e-01	 2.1381670e-01	[ 4.7850188e-01]	 2.2192287e-01


.. parsed-literal::

      21	 5.3064076e-01	 2.0352988e-01	 5.6649730e-01	 2.1282332e-01	[ 6.0274383e-01]	 2.1275353e-01


.. parsed-literal::

      22	 5.7368161e-01	 2.0096831e-01	 6.1428843e-01	 2.0502627e-01	[ 6.6764458e-01]	 2.2115874e-01


.. parsed-literal::

      23	 6.2739812e-01	 1.9652444e-01	 6.6539388e-01	 2.0163873e-01	[ 7.1571037e-01]	 2.0603609e-01


.. parsed-literal::

      24	 6.7164506e-01	 1.9076109e-01	 7.0912751e-01	 1.9679715e-01	[ 7.5720407e-01]	 2.0706367e-01


.. parsed-literal::

      25	 7.1800152e-01	 1.9205915e-01	 7.5424561e-01	 1.9714455e-01	[ 7.9232744e-01]	 2.0801806e-01


.. parsed-literal::

      26	 7.4001567e-01	 1.9414686e-01	 7.7620387e-01	 1.9815292e-01	[ 8.1275682e-01]	 3.3628249e-01


.. parsed-literal::

      27	 7.5951886e-01	 1.9057317e-01	 7.9769483e-01	 1.9449215e-01	[ 8.4883567e-01]	 2.0778227e-01


.. parsed-literal::

      28	 7.9303006e-01	 1.9731591e-01	 8.3090049e-01	 2.0129570e-01	[ 8.6986029e-01]	 2.1093106e-01


.. parsed-literal::

      29	 8.1518427e-01	 1.9741280e-01	 8.5356370e-01	 1.9984494e-01	[ 8.9622636e-01]	 2.0647073e-01
      30	 8.3856133e-01	 1.9321979e-01	 8.7755457e-01	 1.9641025e-01	[ 9.2389087e-01]	 1.8583775e-01


.. parsed-literal::

      31	 8.7049365e-01	 1.8649370e-01	 9.1153761e-01	 1.9883323e-01	[ 9.4628349e-01]	 2.1527052e-01
      32	 8.8451174e-01	 1.8419494e-01	 9.2600772e-01	 1.9267559e-01	[ 9.6512606e-01]	 1.9767261e-01


.. parsed-literal::

      33	 8.9788594e-01	 1.8212360e-01	 9.3889587e-01	 1.9113517e-01	[ 9.7684699e-01]	 2.1295094e-01
      34	 9.1645391e-01	 1.7833010e-01	 9.5742212e-01	 1.8753429e-01	[ 9.9368105e-01]	 1.8694139e-01


.. parsed-literal::

      35	 9.3513742e-01	 1.7533892e-01	 9.7652166e-01	 1.8306368e-01	[ 1.0131823e+00]	 2.0915985e-01


.. parsed-literal::

      36	 9.6330310e-01	 1.7033566e-01	 1.0064376e+00	 1.7839993e-01	[ 1.0345382e+00]	 2.1178508e-01


.. parsed-literal::

      37	 9.6801754e-01	 1.6930297e-01	 1.0119494e+00	 1.7768043e-01	  1.0342631e+00 	 2.0536470e-01


.. parsed-literal::

      38	 9.8270730e-01	 1.6823687e-01	 1.0262514e+00	 1.7727139e-01	[ 1.0464624e+00]	 2.1111941e-01


.. parsed-literal::

      39	 9.9127912e-01	 1.6775431e-01	 1.0348016e+00	 1.7688461e-01	[ 1.0513337e+00]	 2.0289207e-01
      40	 9.9993442e-01	 1.6711086e-01	 1.0437614e+00	 1.7653354e-01	[ 1.0540782e+00]	 1.8244696e-01


.. parsed-literal::

      41	 1.0184678e+00	 1.6592219e-01	 1.0634654e+00	 1.7468245e-01	[ 1.0557525e+00]	 1.9002724e-01


.. parsed-literal::

      42	 1.0275791e+00	 1.6495132e-01	 1.0733560e+00	 1.7480639e-01	[ 1.0668198e+00]	 3.2303572e-01


.. parsed-literal::

      43	 1.0331979e+00	 1.6442926e-01	 1.0791032e+00	 1.7364320e-01	[ 1.0728828e+00]	 2.0446539e-01
      44	 1.0470683e+00	 1.6265846e-01	 1.0935120e+00	 1.7070815e-01	[ 1.0903160e+00]	 1.9961715e-01


.. parsed-literal::

      45	 1.0584284e+00	 1.6134727e-01	 1.1049277e+00	 1.6916997e-01	[ 1.1086947e+00]	 2.0213985e-01


.. parsed-literal::

      46	 1.0742466e+00	 1.5900090e-01	 1.1212833e+00	 1.6781768e-01	[ 1.1377619e+00]	 2.0872402e-01


.. parsed-literal::

      47	 1.0827262e+00	 1.5703288e-01	 1.1293759e+00	 1.6610309e-01	[ 1.1458907e+00]	 2.0679855e-01


.. parsed-literal::

      48	 1.0916173e+00	 1.5629088e-01	 1.1380558e+00	 1.6637070e-01	[ 1.1510293e+00]	 2.0778918e-01


.. parsed-literal::

      49	 1.1033133e+00	 1.5447927e-01	 1.1500390e+00	 1.6546438e-01	[ 1.1599110e+00]	 2.1178818e-01


.. parsed-literal::

      50	 1.1147349e+00	 1.5217192e-01	 1.1616119e+00	 1.6394278e-01	[ 1.1687200e+00]	 2.0289063e-01


.. parsed-literal::

      51	 1.1247371e+00	 1.4901006e-01	 1.1717589e+00	 1.6028063e-01	[ 1.1840903e+00]	 2.0765424e-01


.. parsed-literal::

      52	 1.1343651e+00	 1.4803813e-01	 1.1812709e+00	 1.6020700e-01	[ 1.1923252e+00]	 2.0556283e-01
      53	 1.1476366e+00	 1.4640739e-01	 1.1948071e+00	 1.5939881e-01	[ 1.2053040e+00]	 1.8354201e-01


.. parsed-literal::

      54	 1.1573353e+00	 1.4389710e-01	 1.2050099e+00	 1.5623813e-01	  1.2016548e+00 	 1.8521714e-01
      55	 1.1689579e+00	 1.4257561e-01	 1.2167527e+00	 1.5447909e-01	[ 1.2094502e+00]	 1.7484903e-01


.. parsed-literal::

      56	 1.1775962e+00	 1.4164493e-01	 1.2256501e+00	 1.5325651e-01	[ 1.2118452e+00]	 1.9697475e-01


.. parsed-literal::

      57	 1.1881157e+00	 1.4016127e-01	 1.2367928e+00	 1.5171519e-01	[ 1.2155668e+00]	 2.1299076e-01


.. parsed-literal::

      58	 1.1975703e+00	 1.3910185e-01	 1.2469763e+00	 1.4987433e-01	[ 1.2232203e+00]	 2.1829271e-01


.. parsed-literal::

      59	 1.2060604e+00	 1.3821927e-01	 1.2555145e+00	 1.4816462e-01	[ 1.2359508e+00]	 2.0543432e-01


.. parsed-literal::

      60	 1.2127086e+00	 1.3796281e-01	 1.2619492e+00	 1.4768007e-01	[ 1.2421169e+00]	 2.1928573e-01


.. parsed-literal::

      61	 1.2193855e+00	 1.3770717e-01	 1.2687922e+00	 1.4694914e-01	[ 1.2488026e+00]	 2.1155357e-01


.. parsed-literal::

      62	 1.2263687e+00	 1.3758173e-01	 1.2761641e+00	 1.4661806e-01	  1.2467413e+00 	 2.2002339e-01


.. parsed-literal::

      63	 1.2334543e+00	 1.3715438e-01	 1.2832212e+00	 1.4643444e-01	[ 1.2550175e+00]	 2.0641661e-01


.. parsed-literal::

      64	 1.2416473e+00	 1.3627255e-01	 1.2916220e+00	 1.4620203e-01	[ 1.2596541e+00]	 2.1022606e-01


.. parsed-literal::

      65	 1.2497352e+00	 1.3568168e-01	 1.2998295e+00	 1.4549941e-01	[ 1.2642246e+00]	 2.1711969e-01


.. parsed-literal::

      66	 1.2582985e+00	 1.3487224e-01	 1.3085881e+00	 1.4529621e-01	[ 1.2715144e+00]	 2.1157408e-01


.. parsed-literal::

      67	 1.2658485e+00	 1.3411307e-01	 1.3163257e+00	 1.4418231e-01	[ 1.2802014e+00]	 2.1524167e-01


.. parsed-literal::

      68	 1.2742993e+00	 1.3334645e-01	 1.3252686e+00	 1.4298034e-01	[ 1.2936596e+00]	 2.0849061e-01


.. parsed-literal::

      69	 1.2800925e+00	 1.3200482e-01	 1.3312059e+00	 1.4136148e-01	[ 1.2965836e+00]	 2.0488453e-01
      70	 1.2846779e+00	 1.3213971e-01	 1.3356862e+00	 1.4179146e-01	[ 1.3013811e+00]	 1.9292903e-01


.. parsed-literal::

      71	 1.2899793e+00	 1.3178034e-01	 1.3410893e+00	 1.4215863e-01	[ 1.3051995e+00]	 2.1101570e-01


.. parsed-literal::

      72	 1.2954934e+00	 1.3119072e-01	 1.3467036e+00	 1.4222286e-01	[ 1.3079709e+00]	 2.2123694e-01
      73	 1.3055658e+00	 1.2993102e-01	 1.3570238e+00	 1.4259714e-01	[ 1.3085096e+00]	 1.8768048e-01


.. parsed-literal::

      74	 1.3081944e+00	 1.2856497e-01	 1.3602729e+00	 1.4148538e-01	  1.3004332e+00 	 2.0511794e-01


.. parsed-literal::

      75	 1.3164105e+00	 1.2835380e-01	 1.3679182e+00	 1.4087557e-01	[ 1.3152515e+00]	 2.1337318e-01


.. parsed-literal::

      76	 1.3204728e+00	 1.2787803e-01	 1.3720228e+00	 1.4002767e-01	[ 1.3209793e+00]	 2.1996069e-01


.. parsed-literal::

      77	 1.3255383e+00	 1.2725228e-01	 1.3772113e+00	 1.3961295e-01	[ 1.3229455e+00]	 2.0693183e-01


.. parsed-literal::

      78	 1.3302893e+00	 1.2637841e-01	 1.3823633e+00	 1.3907567e-01	[ 1.3244252e+00]	 2.1190405e-01
      79	 1.3367323e+00	 1.2586740e-01	 1.3885756e+00	 1.3896354e-01	  1.3241894e+00 	 1.9912887e-01


.. parsed-literal::

      80	 1.3397329e+00	 1.2583905e-01	 1.3914733e+00	 1.3929079e-01	[ 1.3252077e+00]	 2.1230936e-01


.. parsed-literal::

      81	 1.3449020e+00	 1.2525889e-01	 1.3969172e+00	 1.3974142e-01	  1.3208339e+00 	 2.0405006e-01


.. parsed-literal::

      82	 1.3470045e+00	 1.2576377e-01	 1.3993524e+00	 1.4041353e-01	  1.3189888e+00 	 2.1113706e-01
      83	 1.3518920e+00	 1.2516265e-01	 1.4041454e+00	 1.3991721e-01	  1.3240018e+00 	 1.7732072e-01


.. parsed-literal::

      84	 1.3553731e+00	 1.2472799e-01	 1.4077124e+00	 1.3964532e-01	[ 1.3264471e+00]	 2.0392013e-01


.. parsed-literal::

      85	 1.3584347e+00	 1.2448196e-01	 1.4108149e+00	 1.3959748e-01	[ 1.3286785e+00]	 2.1085334e-01
      86	 1.3656561e+00	 1.2392515e-01	 1.4180723e+00	 1.3978880e-01	[ 1.3296709e+00]	 1.9882369e-01


.. parsed-literal::

      87	 1.3695204e+00	 1.2382209e-01	 1.4220561e+00	 1.4019855e-01	[ 1.3317151e+00]	 3.2328629e-01


.. parsed-literal::

      88	 1.3748588e+00	 1.2372846e-01	 1.4274721e+00	 1.4079569e-01	  1.3276732e+00 	 2.4730539e-01


.. parsed-literal::

      89	 1.3787576e+00	 1.2356054e-01	 1.4313180e+00	 1.4085759e-01	  1.3293269e+00 	 2.0897579e-01


.. parsed-literal::

      90	 1.3840298e+00	 1.2339728e-01	 1.4366445e+00	 1.4117095e-01	  1.3310654e+00 	 2.1016049e-01
      91	 1.3867390e+00	 1.2287965e-01	 1.4395183e+00	 1.4069850e-01	[ 1.3326150e+00]	 1.9886661e-01


.. parsed-literal::

      92	 1.3902984e+00	 1.2261762e-01	 1.4429705e+00	 1.4082758e-01	[ 1.3399279e+00]	 2.1286774e-01


.. parsed-literal::

      93	 1.3937113e+00	 1.2229961e-01	 1.4463973e+00	 1.4113201e-01	[ 1.3446474e+00]	 2.1313739e-01


.. parsed-literal::

      94	 1.3971521e+00	 1.2197751e-01	 1.4498715e+00	 1.4135027e-01	[ 1.3499522e+00]	 2.1064234e-01


.. parsed-literal::

      95	 1.3992619e+00	 1.2131625e-01	 1.4521589e+00	 1.4256322e-01	  1.3498211e+00 	 2.2414160e-01
      96	 1.4054686e+00	 1.2131592e-01	 1.4581854e+00	 1.4194852e-01	[ 1.3615222e+00]	 1.9987607e-01


.. parsed-literal::

      97	 1.4079769e+00	 1.2132170e-01	 1.4606373e+00	 1.4174344e-01	[ 1.3646087e+00]	 2.1912861e-01


.. parsed-literal::

      98	 1.4113438e+00	 1.2122810e-01	 1.4640438e+00	 1.4172787e-01	[ 1.3674919e+00]	 2.1297407e-01
      99	 1.4152498e+00	 1.2110409e-01	 1.4680120e+00	 1.4154062e-01	[ 1.3716162e+00]	 2.0772958e-01


.. parsed-literal::

     100	 1.4181392e+00	 1.2091193e-01	 1.4710472e+00	 1.4164735e-01	  1.3712462e+00 	 3.2697868e-01


.. parsed-literal::

     101	 1.4212224e+00	 1.2079164e-01	 1.4741724e+00	 1.4187989e-01	[ 1.3729690e+00]	 2.0282102e-01
     102	 1.4238303e+00	 1.2061724e-01	 1.4768009e+00	 1.4175043e-01	[ 1.3741248e+00]	 1.9560742e-01


.. parsed-literal::

     103	 1.4275286e+00	 1.2035406e-01	 1.4806809e+00	 1.4134819e-01	  1.3735391e+00 	 2.0027518e-01
     104	 1.4305364e+00	 1.2008893e-01	 1.4838382e+00	 1.4067120e-01	  1.3666664e+00 	 1.8450832e-01


.. parsed-literal::

     105	 1.4330102e+00	 1.2006287e-01	 1.4862612e+00	 1.4017292e-01	  1.3693265e+00 	 1.8808031e-01


.. parsed-literal::

     106	 1.4363376e+00	 1.2010308e-01	 1.4896862e+00	 1.3934125e-01	  1.3690342e+00 	 2.1641541e-01


.. parsed-literal::

     107	 1.4384936e+00	 1.2004283e-01	 1.4919032e+00	 1.3857590e-01	  1.3688551e+00 	 2.1775007e-01
     108	 1.4411915e+00	 1.1998954e-01	 1.4946728e+00	 1.3810497e-01	  1.3708956e+00 	 1.9539642e-01


.. parsed-literal::

     109	 1.4433546e+00	 1.1999650e-01	 1.4968372e+00	 1.3789949e-01	  1.3682778e+00 	 2.1590590e-01


.. parsed-literal::

     110	 1.4456470e+00	 1.1974840e-01	 1.4991012e+00	 1.3794650e-01	  1.3705057e+00 	 2.0797873e-01


.. parsed-literal::

     111	 1.4478396e+00	 1.1965443e-01	 1.5012695e+00	 1.3791570e-01	  1.3726694e+00 	 2.0779490e-01


.. parsed-literal::

     112	 1.4495719e+00	 1.1945959e-01	 1.5031433e+00	 1.3750762e-01	  1.3720307e+00 	 2.1312332e-01
     113	 1.4521678e+00	 1.1942296e-01	 1.5056426e+00	 1.3770377e-01	[ 1.3760885e+00]	 1.8908906e-01


.. parsed-literal::

     114	 1.4535244e+00	 1.1939904e-01	 1.5070121e+00	 1.3758859e-01	[ 1.3771016e+00]	 2.0486093e-01
     115	 1.4563281e+00	 1.1922818e-01	 1.5099007e+00	 1.3746809e-01	[ 1.3777396e+00]	 2.0621014e-01


.. parsed-literal::

     116	 1.4575479e+00	 1.1901540e-01	 1.5111941e+00	 1.3777046e-01	  1.3766789e+00 	 2.0319963e-01


.. parsed-literal::

     117	 1.4595017e+00	 1.1896813e-01	 1.5130672e+00	 1.3771344e-01	[ 1.3789988e+00]	 2.1801472e-01
     118	 1.4608431e+00	 1.1887838e-01	 1.5143612e+00	 1.3780726e-01	[ 1.3804463e+00]	 1.8823719e-01


.. parsed-literal::

     119	 1.4624257e+00	 1.1877496e-01	 1.5159225e+00	 1.3787234e-01	[ 1.3814344e+00]	 2.1256447e-01


.. parsed-literal::

     120	 1.4651437e+00	 1.1860419e-01	 1.5187269e+00	 1.3788235e-01	[ 1.3815052e+00]	 2.0419598e-01


.. parsed-literal::

     121	 1.4666160e+00	 1.1865310e-01	 1.5202845e+00	 1.3746101e-01	  1.3790084e+00 	 3.3137488e-01


.. parsed-literal::

     122	 1.4686845e+00	 1.1852958e-01	 1.5224688e+00	 1.3746004e-01	  1.3790596e+00 	 2.1872807e-01


.. parsed-literal::

     123	 1.4703078e+00	 1.1847268e-01	 1.5241867e+00	 1.3733783e-01	  1.3785316e+00 	 2.1115565e-01


.. parsed-literal::

     124	 1.4722212e+00	 1.1837783e-01	 1.5262317e+00	 1.3707727e-01	  1.3790848e+00 	 2.1332383e-01


.. parsed-literal::

     125	 1.4731252e+00	 1.1848524e-01	 1.5272850e+00	 1.3696173e-01	  1.3732555e+00 	 2.0938730e-01
     126	 1.4750590e+00	 1.1837655e-01	 1.5291190e+00	 1.3683873e-01	  1.3773649e+00 	 1.8501854e-01


.. parsed-literal::

     127	 1.4761644e+00	 1.1833534e-01	 1.5301560e+00	 1.3679784e-01	  1.3794308e+00 	 1.7468810e-01


.. parsed-literal::

     128	 1.4773792e+00	 1.1832628e-01	 1.5313148e+00	 1.3676381e-01	  1.3796125e+00 	 2.0948243e-01


.. parsed-literal::

     129	 1.4797172e+00	 1.1837448e-01	 1.5336633e+00	 1.3634121e-01	  1.3773234e+00 	 2.1518564e-01


.. parsed-literal::

     130	 1.4805465e+00	 1.1864064e-01	 1.5346100e+00	 1.3618270e-01	  1.3634857e+00 	 2.0136762e-01


.. parsed-literal::

     131	 1.4828553e+00	 1.1858239e-01	 1.5368919e+00	 1.3598854e-01	  1.3688756e+00 	 2.2548985e-01


.. parsed-literal::

     132	 1.4835612e+00	 1.1858638e-01	 1.5376473e+00	 1.3590191e-01	  1.3685606e+00 	 2.1830773e-01


.. parsed-literal::

     133	 1.4849848e+00	 1.1861558e-01	 1.5391487e+00	 1.3576011e-01	  1.3669989e+00 	 2.1897531e-01
     134	 1.4869336e+00	 1.1854550e-01	 1.5412067e+00	 1.3582846e-01	  1.3646431e+00 	 1.9523883e-01


.. parsed-literal::

     135	 1.4890042e+00	 1.1862016e-01	 1.5433261e+00	 1.3558667e-01	  1.3578921e+00 	 2.0131111e-01


.. parsed-literal::

     136	 1.4904802e+00	 1.1852345e-01	 1.5447404e+00	 1.3570345e-01	  1.3600276e+00 	 2.1055031e-01


.. parsed-literal::

     137	 1.4921305e+00	 1.1842537e-01	 1.5463252e+00	 1.3586559e-01	  1.3576458e+00 	 2.0552683e-01


.. parsed-literal::

     138	 1.4934841e+00	 1.1837909e-01	 1.5477226e+00	 1.3584232e-01	  1.3606834e+00 	 2.1537113e-01


.. parsed-literal::

     139	 1.4953451e+00	 1.1840755e-01	 1.5496628e+00	 1.3586998e-01	  1.3576536e+00 	 2.1190000e-01
     140	 1.4968318e+00	 1.1855861e-01	 1.5513784e+00	 1.3563042e-01	  1.3589971e+00 	 1.7629027e-01


.. parsed-literal::

     141	 1.4982094e+00	 1.1862123e-01	 1.5527377e+00	 1.3564704e-01	  1.3586675e+00 	 1.7502880e-01


.. parsed-literal::

     142	 1.4991633e+00	 1.1864257e-01	 1.5536576e+00	 1.3569511e-01	  1.3603821e+00 	 2.1489763e-01


.. parsed-literal::

     143	 1.5014863e+00	 1.1873641e-01	 1.5559744e+00	 1.3582465e-01	  1.3649484e+00 	 2.0715046e-01


.. parsed-literal::

     144	 1.5019656e+00	 1.1891786e-01	 1.5565884e+00	 1.3584360e-01	  1.3633645e+00 	 2.0170546e-01


.. parsed-literal::

     145	 1.5044094e+00	 1.1891130e-01	 1.5589356e+00	 1.3588469e-01	  1.3673776e+00 	 2.0987582e-01


.. parsed-literal::

     146	 1.5053196e+00	 1.1885619e-01	 1.5598476e+00	 1.3590969e-01	  1.3665397e+00 	 2.1088934e-01


.. parsed-literal::

     147	 1.5065289e+00	 1.1877496e-01	 1.5611000e+00	 1.3589984e-01	  1.3633231e+00 	 2.1718073e-01
     148	 1.5079597e+00	 1.1865157e-01	 1.5626108e+00	 1.3596885e-01	  1.3600536e+00 	 1.8685770e-01


.. parsed-literal::

     149	 1.5086432e+00	 1.1864651e-01	 1.5634804e+00	 1.3577171e-01	  1.3473527e+00 	 2.2203255e-01


.. parsed-literal::

     150	 1.5102975e+00	 1.1848127e-01	 1.5650829e+00	 1.3611526e-01	  1.3498740e+00 	 2.0883608e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 7s, sys: 1.3 s, total: 2min 9s
    Wall time: 32.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f34a01fa590>



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
    CPU times: user 1.81 s, sys: 46 ms, total: 1.86 s
    Wall time: 581 ms


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

