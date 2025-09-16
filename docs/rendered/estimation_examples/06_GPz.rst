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
       1	-3.3783435e-01	 3.1918291e-01	-3.2803446e-01	 3.2661126e-01	[-3.4101269e-01]	 4.5904541e-01


.. parsed-literal::

       2	-2.6489974e-01	 3.0712102e-01	-2.4027685e-01	 3.1783584e-01	[-2.6507379e-01]	 2.3008084e-01


.. parsed-literal::

       3	-2.2221827e-01	 2.8839242e-01	-1.8192215e-01	 2.9617626e-01	[-2.0892711e-01]	 2.9679775e-01


.. parsed-literal::

       4	-1.7894423e-01	 2.6949112e-01	-1.3057402e-01	 2.7419329e-01	[-1.5516172e-01]	 3.1933379e-01


.. parsed-literal::

       5	-1.4217985e-01	 2.5641358e-01	-1.1406021e-01	 2.6480577e-01	[-1.5078490e-01]	 2.0831180e-01


.. parsed-literal::

       6	-7.1037019e-02	 2.5274595e-01	-4.2967034e-02	 2.5748179e-01	[-6.1710365e-02]	 2.0649266e-01
       7	-5.0756434e-02	 2.4847484e-01	-2.7715341e-02	 2.5621893e-01	[-5.1276502e-02]	 1.9501901e-01


.. parsed-literal::

       8	-3.9548509e-02	 2.4644747e-01	-1.8820168e-02	 2.5296847e-01	[-4.1431601e-02]	 1.9588137e-01
       9	-2.5743115e-02	 2.4389891e-01	-7.9415181e-03	 2.5088981e-01	[-3.3984171e-02]	 1.8671560e-01


.. parsed-literal::

      10	-1.4160073e-02	 2.4160074e-01	 1.8746342e-03	 2.4900883e-01	[-2.6245143e-02]	 2.1026611e-01


.. parsed-literal::

      11	-9.6110887e-03	 2.4117940e-01	 4.7584594e-03	 2.4989808e-01	[-2.5104570e-02]	 2.1084905e-01


.. parsed-literal::

      12	-4.4506430e-03	 2.3998964e-01	 9.6229868e-03	 2.4829262e-01	[-1.9141031e-02]	 2.0778799e-01


.. parsed-literal::

      13	-2.0606817e-03	 2.3949469e-01	 1.1917054e-02	 2.4749711e-01	[-1.5584513e-02]	 2.0878673e-01


.. parsed-literal::

      14	 3.2470391e-03	 2.3838357e-01	 1.7533461e-02	 2.4545242e-01	[-7.2229312e-03]	 2.0378494e-01


.. parsed-literal::

      15	 2.0587794e-02	 2.3532985e-01	 3.5822094e-02	 2.4035598e-01	[ 1.5722566e-02]	 2.1918011e-01


.. parsed-literal::

      16	 1.2521132e-01	 2.1961169e-01	 1.4572722e-01	 2.2344186e-01	[ 1.4178335e-01]	 3.0731535e-01


.. parsed-literal::

      17	 2.1871161e-01	 2.1757310e-01	 2.4362960e-01	 2.2114272e-01	[ 2.3361590e-01]	 2.1139812e-01


.. parsed-literal::

      18	 2.9180430e-01	 2.1268822e-01	 3.2357542e-01	 2.1550974e-01	[ 3.1654367e-01]	 2.0387292e-01


.. parsed-literal::

      19	 3.2395670e-01	 2.0915640e-01	 3.5510535e-01	 2.1355684e-01	[ 3.3944186e-01]	 2.2121048e-01
      20	 3.7199265e-01	 2.0588197e-01	 4.0323429e-01	 2.1131158e-01	[ 3.8956918e-01]	 1.9372892e-01


.. parsed-literal::

      21	 4.3513427e-01	 2.0249429e-01	 4.6824939e-01	 2.0745247e-01	[ 4.5505440e-01]	 1.9483590e-01


.. parsed-literal::

      22	 5.2993575e-01	 1.9972851e-01	 5.6746368e-01	 2.0220871e-01	[ 5.5188889e-01]	 2.1355438e-01


.. parsed-literal::

      23	 5.8899843e-01	 1.9570104e-01	 6.3014337e-01	 2.0126503e-01	[ 6.0060012e-01]	 3.3238745e-01
      24	 6.2135708e-01	 1.9409357e-01	 6.6318577e-01	 1.9927333e-01	[ 6.1814650e-01]	 2.0392370e-01


.. parsed-literal::

      25	 6.5253885e-01	 1.9162727e-01	 6.9296562e-01	 1.9611180e-01	[ 6.5068231e-01]	 1.8100095e-01


.. parsed-literal::

      26	 6.8834916e-01	 1.8947594e-01	 7.2618235e-01	 1.9228565e-01	[ 6.9758824e-01]	 2.0989418e-01


.. parsed-literal::

      27	 7.1744676e-01	 1.9133495e-01	 7.5376112e-01	 1.9396190e-01	[ 7.3306609e-01]	 2.1030140e-01


.. parsed-literal::

      28	 7.3979469e-01	 1.9583939e-01	 7.7591030e-01	 2.0007527e-01	[ 7.5980416e-01]	 2.0492673e-01
      29	 7.6638037e-01	 1.9597003e-01	 8.0386964e-01	 2.0213314e-01	[ 7.8048295e-01]	 1.9546175e-01


.. parsed-literal::

      30	 7.8882743e-01	 1.9584729e-01	 8.2681627e-01	 2.0317927e-01	[ 8.0569855e-01]	 2.0616531e-01


.. parsed-literal::

      31	 8.0888059e-01	 1.9270467e-01	 8.4716158e-01	 2.0127496e-01	[ 8.2020870e-01]	 2.0416498e-01
      32	 8.3059390e-01	 1.9417487e-01	 8.6869474e-01	 2.0234014e-01	[ 8.5190725e-01]	 2.0127177e-01


.. parsed-literal::

      33	 8.5211884e-01	 1.9228824e-01	 8.9262140e-01	 1.9560934e-01	[ 8.5954701e-01]	 2.0064211e-01
      34	 8.7255903e-01	 1.8944672e-01	 9.1425747e-01	 1.9308249e-01	[ 8.8006712e-01]	 1.8405581e-01


.. parsed-literal::

      35	 8.8603143e-01	 1.8804993e-01	 9.2743954e-01	 1.9137073e-01	[ 9.0072789e-01]	 2.0877767e-01


.. parsed-literal::

      36	 9.0882868e-01	 1.8529972e-01	 9.5034342e-01	 1.8603761e-01	[ 9.3010222e-01]	 2.0443130e-01


.. parsed-literal::

      37	 9.2803416e-01	 1.8412971e-01	 9.6977401e-01	 1.8410676e-01	[ 9.4903983e-01]	 2.1237135e-01


.. parsed-literal::

      38	 9.4719986e-01	 1.8139225e-01	 9.8993408e-01	 1.8104504e-01	[ 9.5534275e-01]	 2.0831490e-01
      39	 9.6110006e-01	 1.7964763e-01	 1.0043587e+00	 1.7984961e-01	[ 9.6701458e-01]	 1.8695593e-01


.. parsed-literal::

      40	 9.8343881e-01	 1.7630479e-01	 1.0276893e+00	 1.7662746e-01	[ 9.9117198e-01]	 1.9273257e-01


.. parsed-literal::

      41	 1.0042608e+00	 1.7134226e-01	 1.0500602e+00	 1.7211549e-01	[ 1.0093310e+00]	 2.1328998e-01


.. parsed-literal::

      42	 1.0250678e+00	 1.6783851e-01	 1.0709028e+00	 1.6875711e-01	[ 1.0310236e+00]	 2.1808863e-01


.. parsed-literal::

      43	 1.0351741e+00	 1.6652646e-01	 1.0812273e+00	 1.6727507e-01	[ 1.0412329e+00]	 2.0091653e-01
      44	 1.0515051e+00	 1.6492326e-01	 1.0980386e+00	 1.6594319e-01	[ 1.0581238e+00]	 2.0153618e-01


.. parsed-literal::

      45	 1.0593406e+00	 1.6207909e-01	 1.1065203e+00	 1.6448960e-01	[ 1.0715703e+00]	 2.0816731e-01


.. parsed-literal::

      46	 1.0740896e+00	 1.6076478e-01	 1.1208718e+00	 1.6306734e-01	[ 1.0815804e+00]	 2.0938444e-01
      47	 1.0840789e+00	 1.5923237e-01	 1.1309134e+00	 1.6180890e-01	[ 1.0879836e+00]	 1.8956184e-01


.. parsed-literal::

      48	 1.0916020e+00	 1.5776740e-01	 1.1384907e+00	 1.6049924e-01	[ 1.0940039e+00]	 1.9554234e-01


.. parsed-literal::

      49	 1.1041434e+00	 1.5583820e-01	 1.1512678e+00	 1.5847136e-01	[ 1.1064715e+00]	 2.1576262e-01


.. parsed-literal::

      50	 1.1140075e+00	 1.5473032e-01	 1.1611585e+00	 1.5813982e-01	[ 1.1166798e+00]	 2.1024108e-01


.. parsed-literal::

      51	 1.1209771e+00	 1.5444251e-01	 1.1679971e+00	 1.5839973e-01	[ 1.1234596e+00]	 2.0700479e-01


.. parsed-literal::

      52	 1.1333848e+00	 1.5399283e-01	 1.1807823e+00	 1.5924850e-01	[ 1.1315989e+00]	 2.0915246e-01


.. parsed-literal::

      53	 1.1444085e+00	 1.5371948e-01	 1.1920021e+00	 1.6064959e-01	[ 1.1398774e+00]	 2.1081209e-01
      54	 1.1571130e+00	 1.5222462e-01	 1.2050598e+00	 1.5915481e-01	[ 1.1500443e+00]	 1.9911289e-01


.. parsed-literal::

      55	 1.1691596e+00	 1.5203480e-01	 1.2182947e+00	 1.5932257e-01	[ 1.1536464e+00]	 1.9019675e-01
      56	 1.1800699e+00	 1.5154579e-01	 1.2292385e+00	 1.5973015e-01	[ 1.1655784e+00]	 1.8390989e-01


.. parsed-literal::

      57	 1.1899060e+00	 1.5090261e-01	 1.2390273e+00	 1.5909102e-01	[ 1.1751585e+00]	 2.0760727e-01


.. parsed-literal::

      58	 1.1998532e+00	 1.5118698e-01	 1.2494699e+00	 1.5958267e-01	[ 1.1833142e+00]	 2.1863413e-01
      59	 1.2085467e+00	 1.5039990e-01	 1.2583900e+00	 1.5927166e-01	[ 1.1863373e+00]	 1.8573880e-01


.. parsed-literal::

      60	 1.2164487e+00	 1.5008308e-01	 1.2661275e+00	 1.5840319e-01	[ 1.1928249e+00]	 2.1128869e-01


.. parsed-literal::

      61	 1.2273188e+00	 1.4932056e-01	 1.2770446e+00	 1.5871273e-01	[ 1.1986340e+00]	 2.1818900e-01
      62	 1.2325514e+00	 1.4928982e-01	 1.2827792e+00	 1.5747214e-01	[ 1.2013543e+00]	 1.9153547e-01


.. parsed-literal::

      63	 1.2408455e+00	 1.4794898e-01	 1.2906938e+00	 1.5830460e-01	[ 1.2098728e+00]	 2.0637274e-01
      64	 1.2457456e+00	 1.4803842e-01	 1.2957331e+00	 1.6013464e-01	[ 1.2134403e+00]	 1.7795253e-01


.. parsed-literal::

      65	 1.2525612e+00	 1.4786378e-01	 1.3027351e+00	 1.6182969e-01	[ 1.2171673e+00]	 2.1534038e-01


.. parsed-literal::

      66	 1.2630982e+00	 1.4671006e-01	 1.3137636e+00	 1.6211809e-01	[ 1.2218707e+00]	 2.0894241e-01
      67	 1.2731751e+00	 1.4574035e-01	 1.3242715e+00	 1.6244960e-01	  1.2218537e+00 	 1.8091130e-01


.. parsed-literal::

      68	 1.2807707e+00	 1.4479967e-01	 1.3317990e+00	 1.6067937e-01	[ 1.2278710e+00]	 2.0435429e-01


.. parsed-literal::

      69	 1.2875489e+00	 1.4355495e-01	 1.3385622e+00	 1.5844484e-01	[ 1.2337785e+00]	 2.1922493e-01


.. parsed-literal::

      70	 1.2938552e+00	 1.4330768e-01	 1.3450856e+00	 1.5821338e-01	[ 1.2404062e+00]	 2.0400596e-01


.. parsed-literal::

      71	 1.3023041e+00	 1.4272697e-01	 1.3537451e+00	 1.5791462e-01	[ 1.2510694e+00]	 2.0957112e-01


.. parsed-literal::

      72	 1.3091552e+00	 1.4207927e-01	 1.3607989e+00	 1.5863680e-01	[ 1.2589998e+00]	 2.1436429e-01
      73	 1.3159237e+00	 1.4177257e-01	 1.3676503e+00	 1.5820104e-01	[ 1.2636466e+00]	 1.9003701e-01


.. parsed-literal::

      74	 1.3208395e+00	 1.4114674e-01	 1.3726807e+00	 1.5763323e-01	[ 1.2637834e+00]	 2.1148419e-01
      75	 1.3270089e+00	 1.4096851e-01	 1.3789698e+00	 1.5687892e-01	[ 1.2667325e+00]	 1.8145728e-01


.. parsed-literal::

      76	 1.3346896e+00	 1.4041987e-01	 1.3869103e+00	 1.5570839e-01	[ 1.2690240e+00]	 2.1582341e-01


.. parsed-literal::

      77	 1.3420207e+00	 1.3984431e-01	 1.3944441e+00	 1.5413921e-01	[ 1.2733746e+00]	 2.1964145e-01


.. parsed-literal::

      78	 1.3484602e+00	 1.3987938e-01	 1.4011542e+00	 1.5391334e-01	[ 1.2810515e+00]	 2.0760536e-01
      79	 1.3525459e+00	 1.3907604e-01	 1.4053248e+00	 1.5176000e-01	[ 1.2836435e+00]	 2.0670319e-01


.. parsed-literal::

      80	 1.3557374e+00	 1.3906739e-01	 1.4084320e+00	 1.5208949e-01	[ 1.2890974e+00]	 2.0733905e-01


.. parsed-literal::

      81	 1.3597973e+00	 1.3884921e-01	 1.4126447e+00	 1.5192174e-01	[ 1.2925276e+00]	 2.0985174e-01


.. parsed-literal::

      82	 1.3640431e+00	 1.3879300e-01	 1.4170742e+00	 1.5143315e-01	[ 1.2967102e+00]	 2.1506548e-01


.. parsed-literal::

      83	 1.3699283e+00	 1.3850405e-01	 1.4231349e+00	 1.5098247e-01	[ 1.2988963e+00]	 2.1871161e-01


.. parsed-literal::

      84	 1.3753527e+00	 1.3839919e-01	 1.4287583e+00	 1.4968034e-01	[ 1.3011732e+00]	 2.0702767e-01


.. parsed-literal::

      85	 1.3798970e+00	 1.3794844e-01	 1.4331828e+00	 1.4962283e-01	[ 1.3035196e+00]	 2.0990181e-01


.. parsed-literal::

      86	 1.3835467e+00	 1.3754120e-01	 1.4368048e+00	 1.4870805e-01	[ 1.3066942e+00]	 2.1395874e-01


.. parsed-literal::

      87	 1.3872835e+00	 1.3721647e-01	 1.4404972e+00	 1.4815498e-01	[ 1.3117311e+00]	 2.0727634e-01
      88	 1.3908620e+00	 1.3680372e-01	 1.4442103e+00	 1.4710334e-01	[ 1.3236201e+00]	 1.7984462e-01


.. parsed-literal::

      89	 1.3949065e+00	 1.3673616e-01	 1.4481242e+00	 1.4687523e-01	[ 1.3266577e+00]	 2.0954466e-01


.. parsed-literal::

      90	 1.3971142e+00	 1.3673412e-01	 1.4503401e+00	 1.4669306e-01	[ 1.3283771e+00]	 2.0435882e-01


.. parsed-literal::

      91	 1.4011071e+00	 1.3647551e-01	 1.4543057e+00	 1.4652000e-01	[ 1.3298078e+00]	 2.0473719e-01


.. parsed-literal::

      92	 1.4050771e+00	 1.3611175e-01	 1.4585150e+00	 1.4650826e-01	  1.3273865e+00 	 2.0132017e-01
      93	 1.4095067e+00	 1.3561158e-01	 1.4628246e+00	 1.4637104e-01	  1.3268199e+00 	 1.8438649e-01


.. parsed-literal::

      94	 1.4118498e+00	 1.3540042e-01	 1.4651536e+00	 1.4615226e-01	  1.3289567e+00 	 2.0021105e-01
      95	 1.4150359e+00	 1.3508900e-01	 1.4684482e+00	 1.4578826e-01	  1.3296601e+00 	 1.8756008e-01


.. parsed-literal::

      96	 1.4175164e+00	 1.3512972e-01	 1.4710518e+00	 1.4603883e-01	[ 1.3313290e+00]	 1.7749596e-01


.. parsed-literal::

      97	 1.4203986e+00	 1.3503104e-01	 1.4739365e+00	 1.4585217e-01	[ 1.3316847e+00]	 2.0160961e-01


.. parsed-literal::

      98	 1.4235913e+00	 1.3496525e-01	 1.4771994e+00	 1.4567696e-01	[ 1.3326983e+00]	 2.1720362e-01


.. parsed-literal::

      99	 1.4259574e+00	 1.3474749e-01	 1.4796088e+00	 1.4570461e-01	  1.3318214e+00 	 2.2019386e-01


.. parsed-literal::

     100	 1.4294258e+00	 1.3429405e-01	 1.4831849e+00	 1.4577734e-01	  1.3308499e+00 	 2.1026778e-01
     101	 1.4324645e+00	 1.3360365e-01	 1.4864974e+00	 1.4633722e-01	  1.3278053e+00 	 1.9104981e-01


.. parsed-literal::

     102	 1.4356258e+00	 1.3332634e-01	 1.4896973e+00	 1.4613409e-01	  1.3296626e+00 	 1.9932127e-01
     103	 1.4380542e+00	 1.3304066e-01	 1.4921258e+00	 1.4584542e-01	  1.3326710e+00 	 1.9239163e-01


.. parsed-literal::

     104	 1.4400071e+00	 1.3260829e-01	 1.4942396e+00	 1.4534032e-01	[ 1.3346769e+00]	 2.0722175e-01


.. parsed-literal::

     105	 1.4421919e+00	 1.3231046e-01	 1.4964380e+00	 1.4532113e-01	[ 1.3371032e+00]	 2.1154404e-01
     106	 1.4438530e+00	 1.3204097e-01	 1.4980917e+00	 1.4511016e-01	[ 1.3390980e+00]	 1.8785691e-01


.. parsed-literal::

     107	 1.4457381e+00	 1.3169719e-01	 1.5000240e+00	 1.4499235e-01	[ 1.3395359e+00]	 1.7805076e-01


.. parsed-literal::

     108	 1.4484742e+00	 1.3112344e-01	 1.5028738e+00	 1.4414377e-01	  1.3380139e+00 	 2.1458459e-01


.. parsed-literal::

     109	 1.4513635e+00	 1.3076693e-01	 1.5057769e+00	 1.4400539e-01	  1.3385693e+00 	 2.1078515e-01
     110	 1.4532438e+00	 1.3064349e-01	 1.5076603e+00	 1.4363686e-01	  1.3388769e+00 	 1.8669868e-01


.. parsed-literal::

     111	 1.4551879e+00	 1.3055084e-01	 1.5096475e+00	 1.4329512e-01	  1.3385681e+00 	 1.9477391e-01
     112	 1.4565804e+00	 1.3065044e-01	 1.5111195e+00	 1.4283578e-01	  1.3379527e+00 	 1.8407488e-01


.. parsed-literal::

     113	 1.4589672e+00	 1.3061808e-01	 1.5134332e+00	 1.4300209e-01	[ 1.3398614e+00]	 2.0567679e-01
     114	 1.4603908e+00	 1.3058217e-01	 1.5148223e+00	 1.4323450e-01	[ 1.3417625e+00]	 1.9332957e-01


.. parsed-literal::

     115	 1.4618129e+00	 1.3058547e-01	 1.5162150e+00	 1.4346258e-01	[ 1.3438399e+00]	 2.0352554e-01


.. parsed-literal::

     116	 1.4642803e+00	 1.3044611e-01	 1.5187691e+00	 1.4376306e-01	[ 1.3468161e+00]	 2.1414661e-01


.. parsed-literal::

     117	 1.4664436e+00	 1.3053504e-01	 1.5211032e+00	 1.4400576e-01	[ 1.3475119e+00]	 2.1406841e-01


.. parsed-literal::

     118	 1.4684549e+00	 1.3036977e-01	 1.5231089e+00	 1.4392632e-01	[ 1.3478123e+00]	 2.0486403e-01


.. parsed-literal::

     119	 1.4699263e+00	 1.3024691e-01	 1.5246559e+00	 1.4378855e-01	  1.3473174e+00 	 2.0621371e-01


.. parsed-literal::

     120	 1.4717986e+00	 1.3012264e-01	 1.5265831e+00	 1.4363379e-01	[ 1.3482546e+00]	 2.1975756e-01


.. parsed-literal::

     121	 1.4730744e+00	 1.3012318e-01	 1.5280950e+00	 1.4427838e-01	[ 1.3493266e+00]	 2.0421886e-01


.. parsed-literal::

     122	 1.4758691e+00	 1.3011896e-01	 1.5307717e+00	 1.4407906e-01	[ 1.3507921e+00]	 2.0906758e-01
     123	 1.4767800e+00	 1.3016187e-01	 1.5316419e+00	 1.4418141e-01	[ 1.3512225e+00]	 1.9915056e-01


.. parsed-literal::

     124	 1.4785759e+00	 1.3022461e-01	 1.5334820e+00	 1.4455696e-01	  1.3498678e+00 	 1.9939661e-01


.. parsed-literal::

     125	 1.4795977e+00	 1.3029571e-01	 1.5347089e+00	 1.4515764e-01	  1.3419508e+00 	 2.1560884e-01
     126	 1.4815774e+00	 1.3022803e-01	 1.5366690e+00	 1.4530269e-01	  1.3429462e+00 	 1.9084764e-01


.. parsed-literal::

     127	 1.4827197e+00	 1.3016638e-01	 1.5378651e+00	 1.4538706e-01	  1.3417326e+00 	 1.9637179e-01


.. parsed-literal::

     128	 1.4841779e+00	 1.3008092e-01	 1.5393879e+00	 1.4550457e-01	  1.3386270e+00 	 2.1715426e-01


.. parsed-literal::

     129	 1.4858224e+00	 1.3008741e-01	 1.5410696e+00	 1.4558121e-01	  1.3384905e+00 	 2.1537209e-01


.. parsed-literal::

     130	 1.4877678e+00	 1.2998690e-01	 1.5429946e+00	 1.4563882e-01	  1.3349405e+00 	 2.0666647e-01


.. parsed-literal::

     131	 1.4886924e+00	 1.2994639e-01	 1.5438606e+00	 1.4556559e-01	  1.3353902e+00 	 2.0757508e-01


.. parsed-literal::

     132	 1.4899053e+00	 1.2992381e-01	 1.5450382e+00	 1.4558740e-01	  1.3359868e+00 	 2.1337461e-01


.. parsed-literal::

     133	 1.4915716e+00	 1.2979718e-01	 1.5467481e+00	 1.4594423e-01	  1.3351347e+00 	 2.0684505e-01
     134	 1.4931352e+00	 1.2977651e-01	 1.5483348e+00	 1.4612685e-01	  1.3347531e+00 	 2.0119166e-01


.. parsed-literal::

     135	 1.4942965e+00	 1.2977260e-01	 1.5495619e+00	 1.4642135e-01	  1.3331728e+00 	 2.0406246e-01
     136	 1.4956942e+00	 1.2974996e-01	 1.5510343e+00	 1.4672566e-01	  1.3303589e+00 	 1.9959807e-01


.. parsed-literal::

     137	 1.4969407e+00	 1.2973748e-01	 1.5523213e+00	 1.4683475e-01	  1.3295076e+00 	 2.0151544e-01


.. parsed-literal::

     138	 1.4982683e+00	 1.2964174e-01	 1.5536119e+00	 1.4676044e-01	  1.3300972e+00 	 2.0708203e-01
     139	 1.4998747e+00	 1.2951743e-01	 1.5551407e+00	 1.4662262e-01	  1.3316957e+00 	 1.7306328e-01


.. parsed-literal::

     140	 1.5010407e+00	 1.2943722e-01	 1.5562964e+00	 1.4666024e-01	  1.3314845e+00 	 1.8438721e-01


.. parsed-literal::

     141	 1.5027358e+00	 1.2937438e-01	 1.5580062e+00	 1.4720376e-01	  1.3312556e+00 	 2.1336007e-01
     142	 1.5039756e+00	 1.2938127e-01	 1.5593002e+00	 1.4763890e-01	  1.3284558e+00 	 1.9384837e-01


.. parsed-literal::

     143	 1.5051833e+00	 1.2935505e-01	 1.5605627e+00	 1.4815853e-01	  1.3279594e+00 	 1.7277646e-01


.. parsed-literal::

     144	 1.5064969e+00	 1.2933162e-01	 1.5619417e+00	 1.4868966e-01	  1.3260567e+00 	 2.0351982e-01


.. parsed-literal::

     145	 1.5076687e+00	 1.2927333e-01	 1.5631876e+00	 1.4912201e-01	  1.3211376e+00 	 2.0801973e-01


.. parsed-literal::

     146	 1.5089188e+00	 1.2916539e-01	 1.5644465e+00	 1.4929484e-01	  1.3217186e+00 	 2.1090436e-01
     147	 1.5098288e+00	 1.2906087e-01	 1.5653056e+00	 1.4913421e-01	  1.3220522e+00 	 1.9872165e-01


.. parsed-literal::

     148	 1.5108275e+00	 1.2901974e-01	 1.5662461e+00	 1.4893954e-01	  1.3226788e+00 	 2.1005917e-01


.. parsed-literal::

     149	 1.5118300e+00	 1.2895328e-01	 1.5672222e+00	 1.4880856e-01	  1.3220476e+00 	 2.1183896e-01


.. parsed-literal::

     150	 1.5128247e+00	 1.2895918e-01	 1.5682285e+00	 1.4885934e-01	  1.3205260e+00 	 2.1466899e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 4s, sys: 1.08 s, total: 2min 5s
    Wall time: 31.5 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fddb4d34ac0>



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
    CPU times: user 2.02 s, sys: 50 ms, total: 2.07 s
    Wall time: 619 ms


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

