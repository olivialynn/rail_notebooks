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
       1	-3.2676222e-01	 3.1597018e-01	-3.1701736e-01	 3.3833221e-01	[-3.6147565e-01]	 4.6532011e-01


.. parsed-literal::

       2	-2.5976863e-01	 3.0621944e-01	-2.3649363e-01	 3.2707895e-01	[-3.0327147e-01]	 2.3012900e-01


.. parsed-literal::

       3	-2.1482159e-01	 2.8513563e-01	-1.7254063e-01	 3.0510344e-01	[-2.5979189e-01]	 2.9425859e-01
       4	-1.8874747e-01	 2.6257068e-01	-1.4687146e-01	 2.8368053e-01	 -2.6856437e-01 	 1.8285894e-01


.. parsed-literal::

       5	-9.7628582e-02	 2.5543706e-01	-6.2859490e-02	 2.7225380e-01	[-1.4220355e-01]	 2.1765876e-01


.. parsed-literal::

       6	-6.3406720e-02	 2.4947958e-01	-3.1979331e-02	 2.6516830e-01	[-9.0083548e-02]	 2.1331477e-01


.. parsed-literal::

       7	-4.5028828e-02	 2.4684945e-01	-2.0539804e-02	 2.6176516e-01	[-7.9606774e-02]	 2.1056890e-01


.. parsed-literal::

       8	-3.0914988e-02	 2.4445786e-01	-1.0493504e-02	 2.5901585e-01	[-7.0100248e-02]	 2.1019769e-01
       9	-1.6017355e-02	 2.4168900e-01	 1.3776942e-03	 2.5588622e-01	[-5.8490143e-02]	 1.8873668e-01


.. parsed-literal::

      10	-5.6815594e-03	 2.3986820e-01	 9.4816228e-03	 2.5235224e-01	[-4.4403780e-02]	 2.0786238e-01


.. parsed-literal::

      11	-7.6867090e-04	 2.3912527e-01	 1.3320602e-02	 2.5088570e-01	[-3.7348854e-02]	 2.1067786e-01
      12	 1.8427110e-03	 2.3859797e-01	 1.5716087e-02	 2.5018366e-01	[-3.2999319e-02]	 1.7636061e-01


.. parsed-literal::

      13	 5.1959432e-03	 2.3803672e-01	 1.8880040e-02	 2.4949243e-01	[-3.0142368e-02]	 2.0914817e-01


.. parsed-literal::

      14	 1.1793834e-02	 2.3672687e-01	 2.6201889e-02	 2.4845158e-01	[-2.1366876e-02]	 2.1158314e-01
      15	 7.4084267e-02	 2.2735854e-01	 9.2269374e-02	 2.3630427e-01	[ 6.1786601e-02]	 1.8446827e-01


.. parsed-literal::

      16	 9.2286076e-02	 2.2456036e-01	 1.1091979e-01	 2.3687702e-01	[ 8.2068312e-02]	 2.9606843e-01
      17	 1.4633063e-01	 2.2020014e-01	 1.6782054e-01	 2.3558912e-01	[ 1.3648548e-01]	 1.8117881e-01


.. parsed-literal::

      18	 2.6063462e-01	 2.1999934e-01	 2.9097593e-01	 2.2950619e-01	[ 2.4756072e-01]	 2.0776486e-01


.. parsed-literal::

      19	 3.2064515e-01	 2.1354015e-01	 3.5174816e-01	 2.2357721e-01	[ 3.0784609e-01]	 2.0792317e-01
      20	 3.8671651e-01	 2.0661584e-01	 4.1824793e-01	 2.1931165e-01	[ 3.8323225e-01]	 1.9745159e-01


.. parsed-literal::

      21	 4.4319582e-01	 2.0281556e-01	 4.7530043e-01	 2.1194191e-01	[ 4.4462956e-01]	 2.0058274e-01
      22	 5.2892022e-01	 2.0086004e-01	 5.6335330e-01	 2.0729272e-01	[ 5.3684935e-01]	 2.0247102e-01


.. parsed-literal::

      23	 5.9211086e-01	 1.9999839e-01	 6.3118070e-01	 2.0829446e-01	[ 6.1319606e-01]	 2.1167040e-01
      24	 6.3601447e-01	 1.9517843e-01	 6.7540917e-01	 2.0253158e-01	[ 6.5182799e-01]	 1.8903947e-01


.. parsed-literal::

      25	 6.6012513e-01	 1.9183347e-01	 6.9845637e-01	 1.9835292e-01	[ 6.7335993e-01]	 2.1209121e-01
      26	 6.9857836e-01	 1.9098683e-01	 7.3467675e-01	 2.0070860e-01	[ 7.0053003e-01]	 1.8332338e-01


.. parsed-literal::

      27	 7.3635418e-01	 1.8799219e-01	 7.7305963e-01	 1.9612028e-01	[ 7.4451564e-01]	 2.1390963e-01


.. parsed-literal::

      28	 7.7245613e-01	 1.8426512e-01	 8.1107794e-01	 1.9346565e-01	[ 7.7933875e-01]	 2.1834564e-01
      29	 7.9704308e-01	 1.8365700e-01	 8.3701806e-01	 1.9270821e-01	[ 7.9670309e-01]	 1.8056393e-01


.. parsed-literal::

      30	 8.2190301e-01	 1.8082914e-01	 8.6152322e-01	 1.8766726e-01	[ 8.2361374e-01]	 2.0816302e-01
      31	 8.3869523e-01	 1.7898366e-01	 8.7804247e-01	 1.8444148e-01	[ 8.4167358e-01]	 1.9515967e-01


.. parsed-literal::

      32	 8.7634263e-01	 1.7600541e-01	 9.1616094e-01	 1.7930155e-01	[ 8.7336102e-01]	 2.0843935e-01


.. parsed-literal::

      33	 8.9835803e-01	 1.7243887e-01	 9.3895383e-01	 1.7674303e-01	[ 8.8797530e-01]	 2.0808530e-01


.. parsed-literal::

      34	 9.1745386e-01	 1.6961118e-01	 9.5834388e-01	 1.7458299e-01	[ 9.0528703e-01]	 2.0987153e-01
      35	 9.2863268e-01	 1.6768838e-01	 9.6978799e-01	 1.7198039e-01	[ 9.1442666e-01]	 2.0182991e-01


.. parsed-literal::

      36	 9.4794281e-01	 1.6575241e-01	 9.8989959e-01	 1.6953307e-01	[ 9.2777784e-01]	 2.0697165e-01


.. parsed-literal::

      37	 9.6435043e-01	 1.6529085e-01	 1.0069193e+00	 1.6864009e-01	[ 9.4136641e-01]	 2.0449328e-01


.. parsed-literal::

      38	 9.7890258e-01	 1.6361766e-01	 1.0217785e+00	 1.6602135e-01	[ 9.5860087e-01]	 2.2949934e-01
      39	 9.9327197e-01	 1.6270515e-01	 1.0368167e+00	 1.6491561e-01	[ 9.7431924e-01]	 1.9092965e-01


.. parsed-literal::

      40	 1.0089774e+00	 1.6315562e-01	 1.0534420e+00	 1.6453986e-01	[ 9.8899661e-01]	 2.1437120e-01


.. parsed-literal::

      41	 1.0210186e+00	 1.6652096e-01	 1.0668490e+00	 1.6943477e-01	[ 9.9771680e-01]	 2.0760846e-01


.. parsed-literal::

      42	 1.0321616e+00	 1.6431027e-01	 1.0777739e+00	 1.6562701e-01	[ 1.0103417e+00]	 2.1270990e-01
      43	 1.0407063e+00	 1.6297459e-01	 1.0864386e+00	 1.6332074e-01	[ 1.0196303e+00]	 1.9262099e-01


.. parsed-literal::

      44	 1.0507170e+00	 1.6188737e-01	 1.0966953e+00	 1.6189349e-01	[ 1.0311783e+00]	 1.7697501e-01


.. parsed-literal::

      45	 1.0662580e+00	 1.6154414e-01	 1.1131644e+00	 1.6039427e-01	[ 1.0548348e+00]	 2.0846248e-01


.. parsed-literal::

      46	 1.0792670e+00	 1.5961749e-01	 1.1265168e+00	 1.5942362e-01	[ 1.0717744e+00]	 2.0593572e-01


.. parsed-literal::

      47	 1.0874674e+00	 1.5898140e-01	 1.1345185e+00	 1.5887568e-01	[ 1.0817228e+00]	 2.0661473e-01


.. parsed-literal::

      48	 1.1036559e+00	 1.5805371e-01	 1.1510435e+00	 1.5781999e-01	[ 1.1047020e+00]	 2.1336937e-01


.. parsed-literal::

      49	 1.1093433e+00	 1.5688439e-01	 1.1570146e+00	 1.5698303e-01	[ 1.1092106e+00]	 2.1158671e-01
      50	 1.1190921e+00	 1.5673350e-01	 1.1666535e+00	 1.5714165e-01	[ 1.1191024e+00]	 1.7698836e-01


.. parsed-literal::

      51	 1.1264734e+00	 1.5649440e-01	 1.1740852e+00	 1.5730589e-01	[ 1.1241704e+00]	 2.0296931e-01


.. parsed-literal::

      52	 1.1343004e+00	 1.5617603e-01	 1.1820159e+00	 1.5715544e-01	[ 1.1307439e+00]	 2.1949863e-01
      53	 1.1493751e+00	 1.5530003e-01	 1.1973069e+00	 1.5613115e-01	[ 1.1445941e+00]	 1.8876362e-01


.. parsed-literal::

      54	 1.1524032e+00	 1.5541020e-01	 1.2010021e+00	 1.5803032e-01	[ 1.1524983e+00]	 1.8718481e-01


.. parsed-literal::

      55	 1.1661144e+00	 1.5395041e-01	 1.2142702e+00	 1.5499438e-01	[ 1.1618786e+00]	 2.1108460e-01


.. parsed-literal::

      56	 1.1719436e+00	 1.5343915e-01	 1.2200984e+00	 1.5380029e-01	[ 1.1687094e+00]	 2.1264744e-01


.. parsed-literal::

      57	 1.1785690e+00	 1.5288123e-01	 1.2268703e+00	 1.5286615e-01	[ 1.1758068e+00]	 2.1370578e-01


.. parsed-literal::

      58	 1.1920900e+00	 1.5233825e-01	 1.2406902e+00	 1.5158644e-01	[ 1.1883989e+00]	 2.0532370e-01


.. parsed-literal::

      59	 1.2051388e+00	 1.5134696e-01	 1.2542713e+00	 1.4996279e-01	[ 1.1996895e+00]	 2.1594262e-01
      60	 1.2156519e+00	 1.5084297e-01	 1.2650084e+00	 1.4908274e-01	[ 1.2047519e+00]	 1.7496467e-01


.. parsed-literal::

      61	 1.2275003e+00	 1.5020261e-01	 1.2772522e+00	 1.4775536e-01	[ 1.2094208e+00]	 2.0827198e-01


.. parsed-literal::

      62	 1.2351766e+00	 1.4960723e-01	 1.2850622e+00	 1.4767161e-01	  1.2091843e+00 	 2.0749402e-01


.. parsed-literal::

      63	 1.2417460e+00	 1.4927023e-01	 1.2916116e+00	 1.4733512e-01	[ 1.2142342e+00]	 2.1033740e-01


.. parsed-literal::

      64	 1.2520390e+00	 1.4864237e-01	 1.3022724e+00	 1.4700065e-01	[ 1.2218272e+00]	 2.0824409e-01


.. parsed-literal::

      65	 1.2605189e+00	 1.4820726e-01	 1.3108612e+00	 1.4659799e-01	[ 1.2311651e+00]	 2.1030760e-01
      66	 1.2687230e+00	 1.4814066e-01	 1.3195665e+00	 1.4645370e-01	[ 1.2446072e+00]	 1.7502022e-01


.. parsed-literal::

      67	 1.2780632e+00	 1.4755431e-01	 1.3285626e+00	 1.4543699e-01	[ 1.2546495e+00]	 2.2016215e-01
      68	 1.2835445e+00	 1.4744598e-01	 1.3339651e+00	 1.4506966e-01	[ 1.2607118e+00]	 1.7904305e-01


.. parsed-literal::

      69	 1.2920989e+00	 1.4731244e-01	 1.3428044e+00	 1.4457835e-01	[ 1.2669630e+00]	 2.0738864e-01


.. parsed-literal::

      70	 1.2984942e+00	 1.4723025e-01	 1.3493092e+00	 1.4386082e-01	[ 1.2675194e+00]	 2.1447587e-01
      71	 1.3052432e+00	 1.4680663e-01	 1.3560041e+00	 1.4354206e-01	[ 1.2752027e+00]	 1.9723630e-01


.. parsed-literal::

      72	 1.3117063e+00	 1.4623243e-01	 1.3625953e+00	 1.4316346e-01	[ 1.2795169e+00]	 2.1388173e-01


.. parsed-literal::

      73	 1.3184553e+00	 1.4580360e-01	 1.3695932e+00	 1.4289343e-01	[ 1.2835831e+00]	 2.1266794e-01
      74	 1.3268349e+00	 1.4536680e-01	 1.3781143e+00	 1.4287717e-01	[ 1.2852329e+00]	 1.7896819e-01


.. parsed-literal::

      75	 1.3342809e+00	 1.4519497e-01	 1.3856167e+00	 1.4260508e-01	[ 1.2929315e+00]	 1.9518137e-01


.. parsed-literal::

      76	 1.3396369e+00	 1.4516389e-01	 1.3910930e+00	 1.4227328e-01	[ 1.2962716e+00]	 2.1166301e-01


.. parsed-literal::

      77	 1.3473919e+00	 1.4480044e-01	 1.3991171e+00	 1.4123217e-01	[ 1.3014239e+00]	 2.1644402e-01
      78	 1.3486581e+00	 1.4512588e-01	 1.4006592e+00	 1.4108674e-01	  1.2956920e+00 	 1.8563390e-01


.. parsed-literal::

      79	 1.3579850e+00	 1.4455741e-01	 1.4097539e+00	 1.4050765e-01	[ 1.3080192e+00]	 2.0571375e-01


.. parsed-literal::

      80	 1.3611437e+00	 1.4426006e-01	 1.4129054e+00	 1.4030703e-01	[ 1.3123908e+00]	 2.1387935e-01
      81	 1.3662973e+00	 1.4394251e-01	 1.4182359e+00	 1.4005690e-01	[ 1.3149278e+00]	 1.9323444e-01


.. parsed-literal::

      82	 1.3707336e+00	 1.4366276e-01	 1.4228951e+00	 1.4007050e-01	  1.3139225e+00 	 2.0959592e-01
      83	 1.3764649e+00	 1.4356125e-01	 1.4287026e+00	 1.3969613e-01	[ 1.3159525e+00]	 2.0126081e-01


.. parsed-literal::

      84	 1.3810367e+00	 1.4361594e-01	 1.4332822e+00	 1.3957856e-01	[ 1.3172218e+00]	 2.0774150e-01
      85	 1.3855931e+00	 1.4367862e-01	 1.4379443e+00	 1.3951672e-01	[ 1.3196858e+00]	 1.9644117e-01


.. parsed-literal::

      86	 1.3878439e+00	 1.4367091e-01	 1.4405187e+00	 1.3978551e-01	[ 1.3258672e+00]	 2.1598411e-01
      87	 1.3931279e+00	 1.4341592e-01	 1.4456241e+00	 1.3955458e-01	[ 1.3319635e+00]	 1.7676711e-01


.. parsed-literal::

      88	 1.3954815e+00	 1.4318097e-01	 1.4479866e+00	 1.3941391e-01	[ 1.3356694e+00]	 2.0907259e-01


.. parsed-literal::

      89	 1.3993761e+00	 1.4278189e-01	 1.4519645e+00	 1.3928043e-01	[ 1.3430701e+00]	 2.0861983e-01
      90	 1.4044994e+00	 1.4228495e-01	 1.4572355e+00	 1.3936573e-01	[ 1.3502249e+00]	 1.9547939e-01


.. parsed-literal::

      91	 1.4093715e+00	 1.4182304e-01	 1.4622967e+00	 1.3941884e-01	[ 1.3586343e+00]	 2.1719956e-01


.. parsed-literal::

      92	 1.4127010e+00	 1.4183737e-01	 1.4655694e+00	 1.3942769e-01	[ 1.3613006e+00]	 2.1077085e-01
      93	 1.4161640e+00	 1.4180981e-01	 1.4689816e+00	 1.3920324e-01	[ 1.3629209e+00]	 1.8276000e-01


.. parsed-literal::

      94	 1.4195659e+00	 1.4153979e-01	 1.4724108e+00	 1.3891635e-01	[ 1.3670792e+00]	 1.9964671e-01
      95	 1.4233006e+00	 1.4127140e-01	 1.4761877e+00	 1.3864995e-01	[ 1.3715675e+00]	 1.9904017e-01


.. parsed-literal::

      96	 1.4277232e+00	 1.4084152e-01	 1.4807302e+00	 1.3866550e-01	[ 1.3752429e+00]	 2.0855975e-01


.. parsed-literal::

      97	 1.4308475e+00	 1.4062620e-01	 1.4840077e+00	 1.3873347e-01	[ 1.3784493e+00]	 2.1117496e-01


.. parsed-literal::

      98	 1.4341787e+00	 1.4057789e-01	 1.4872713e+00	 1.3902135e-01	[ 1.3806299e+00]	 2.1308088e-01


.. parsed-literal::

      99	 1.4382890e+00	 1.4055829e-01	 1.4914551e+00	 1.3969668e-01	  1.3799254e+00 	 2.1387267e-01


.. parsed-literal::

     100	 1.4407566e+00	 1.4050531e-01	 1.4939342e+00	 1.3968663e-01	[ 1.3820824e+00]	 2.0967531e-01


.. parsed-literal::

     101	 1.4451334e+00	 1.4041837e-01	 1.4984615e+00	 1.4003397e-01	  1.3805961e+00 	 2.1649146e-01


.. parsed-literal::

     102	 1.4488496e+00	 1.4012791e-01	 1.5023406e+00	 1.3960166e-01	  1.3763438e+00 	 2.0987988e-01


.. parsed-literal::

     103	 1.4521947e+00	 1.3987991e-01	 1.5057285e+00	 1.3916894e-01	  1.3768294e+00 	 2.1512961e-01
     104	 1.4550683e+00	 1.3951344e-01	 1.5086658e+00	 1.3878983e-01	  1.3776465e+00 	 1.7061949e-01


.. parsed-literal::

     105	 1.4585099e+00	 1.3913903e-01	 1.5121949e+00	 1.3847029e-01	  1.3746249e+00 	 1.7521930e-01


.. parsed-literal::

     106	 1.4615319e+00	 1.3876368e-01	 1.5152872e+00	 1.3825659e-01	  1.3758616e+00 	 2.1324468e-01


.. parsed-literal::

     107	 1.4641734e+00	 1.3870126e-01	 1.5178818e+00	 1.3831119e-01	  1.3760494e+00 	 2.1646643e-01


.. parsed-literal::

     108	 1.4666009e+00	 1.3864638e-01	 1.5202956e+00	 1.3823443e-01	  1.3766329e+00 	 2.0756030e-01
     109	 1.4688799e+00	 1.3858677e-01	 1.5226371e+00	 1.3826386e-01	  1.3747136e+00 	 1.7210126e-01


.. parsed-literal::

     110	 1.4720280e+00	 1.3839675e-01	 1.5259332e+00	 1.3822939e-01	  1.3691321e+00 	 1.9643331e-01


.. parsed-literal::

     111	 1.4750515e+00	 1.3839119e-01	 1.5289719e+00	 1.3821514e-01	  1.3741459e+00 	 2.1060348e-01
     112	 1.4783973e+00	 1.3835581e-01	 1.5323915e+00	 1.3843913e-01	  1.3762024e+00 	 1.9969511e-01


.. parsed-literal::

     113	 1.4810402e+00	 1.3827899e-01	 1.5350702e+00	 1.3840539e-01	  1.3808923e+00 	 2.1071506e-01


.. parsed-literal::

     114	 1.4836440e+00	 1.3834360e-01	 1.5376687e+00	 1.3863604e-01	[ 1.3862600e+00]	 2.1679783e-01


.. parsed-literal::

     115	 1.4864718e+00	 1.3822978e-01	 1.5405698e+00	 1.3865857e-01	[ 1.3890396e+00]	 2.1729183e-01


.. parsed-literal::

     116	 1.4887318e+00	 1.3825311e-01	 1.5428945e+00	 1.3868534e-01	[ 1.3951157e+00]	 2.1171284e-01


.. parsed-literal::

     117	 1.4908871e+00	 1.3811398e-01	 1.5450377e+00	 1.3844250e-01	[ 1.3955207e+00]	 2.1814179e-01


.. parsed-literal::

     118	 1.4934661e+00	 1.3792695e-01	 1.5477146e+00	 1.3818112e-01	  1.3953548e+00 	 2.0586133e-01
     119	 1.4950887e+00	 1.3779797e-01	 1.5493923e+00	 1.3800449e-01	  1.3924392e+00 	 1.7681670e-01


.. parsed-literal::

     120	 1.4967888e+00	 1.3781180e-01	 1.5511007e+00	 1.3813563e-01	  1.3903854e+00 	 2.1386051e-01
     121	 1.4995052e+00	 1.3781780e-01	 1.5538585e+00	 1.3840693e-01	  1.3864860e+00 	 1.7704177e-01


.. parsed-literal::

     122	 1.5010371e+00	 1.3791427e-01	 1.5553904e+00	 1.3854104e-01	  1.3832381e+00 	 2.1303415e-01


.. parsed-literal::

     123	 1.5029630e+00	 1.3789390e-01	 1.5573067e+00	 1.3854349e-01	  1.3849945e+00 	 2.1107984e-01


.. parsed-literal::

     124	 1.5058225e+00	 1.3765732e-01	 1.5602633e+00	 1.3811881e-01	  1.3854929e+00 	 2.0779037e-01


.. parsed-literal::

     125	 1.5073417e+00	 1.3772558e-01	 1.5617761e+00	 1.3836741e-01	  1.3910424e+00 	 2.0448804e-01


.. parsed-literal::

     126	 1.5087951e+00	 1.3764233e-01	 1.5631448e+00	 1.3820612e-01	  1.3945834e+00 	 2.1308899e-01
     127	 1.5099335e+00	 1.3755502e-01	 1.5642801e+00	 1.3813087e-01	  1.3953491e+00 	 2.0163488e-01


.. parsed-literal::

     128	 1.5116088e+00	 1.3741390e-01	 1.5659848e+00	 1.3803963e-01	[ 1.3964154e+00]	 2.0087624e-01


.. parsed-literal::

     129	 1.5132111e+00	 1.3721432e-01	 1.5676872e+00	 1.3796308e-01	[ 1.3993617e+00]	 2.0241117e-01


.. parsed-literal::

     130	 1.5153358e+00	 1.3704033e-01	 1.5698150e+00	 1.3784320e-01	[ 1.4000126e+00]	 2.1162271e-01


.. parsed-literal::

     131	 1.5163746e+00	 1.3696942e-01	 1.5708521e+00	 1.3774456e-01	[ 1.4007261e+00]	 2.0438910e-01


.. parsed-literal::

     132	 1.5179505e+00	 1.3682861e-01	 1.5724677e+00	 1.3750129e-01	[ 1.4018562e+00]	 2.0899367e-01
     133	 1.5200178e+00	 1.3645876e-01	 1.5746881e+00	 1.3677282e-01	[ 1.4027854e+00]	 1.9624472e-01


.. parsed-literal::

     134	 1.5222012e+00	 1.3626385e-01	 1.5769486e+00	 1.3638799e-01	  1.4004869e+00 	 2.1230721e-01


.. parsed-literal::

     135	 1.5238203e+00	 1.3612688e-01	 1.5785970e+00	 1.3616371e-01	  1.3996090e+00 	 2.0466709e-01


.. parsed-literal::

     136	 1.5254523e+00	 1.3595419e-01	 1.5802893e+00	 1.3591038e-01	  1.3969844e+00 	 2.1039486e-01


.. parsed-literal::

     137	 1.5269969e+00	 1.3566630e-01	 1.5819201e+00	 1.3567735e-01	  1.3934319e+00 	 2.1202374e-01


.. parsed-literal::

     138	 1.5288486e+00	 1.3535652e-01	 1.5837955e+00	 1.3535710e-01	  1.3908001e+00 	 2.0728660e-01
     139	 1.5305799e+00	 1.3493080e-01	 1.5855791e+00	 1.3505177e-01	  1.3874369e+00 	 1.9977736e-01


.. parsed-literal::

     140	 1.5317679e+00	 1.3473997e-01	 1.5867142e+00	 1.3485714e-01	  1.3896082e+00 	 2.0797896e-01
     141	 1.5329318e+00	 1.3453062e-01	 1.5878407e+00	 1.3467601e-01	  1.3918976e+00 	 1.9863009e-01


.. parsed-literal::

     142	 1.5347581e+00	 1.3408130e-01	 1.5896570e+00	 1.3432684e-01	  1.3953215e+00 	 1.7788315e-01


.. parsed-literal::

     143	 1.5358394e+00	 1.3378449e-01	 1.5907476e+00	 1.3415662e-01	  1.3965922e+00 	 3.2288194e-01
     144	 1.5374039e+00	 1.3343580e-01	 1.5923427e+00	 1.3394496e-01	  1.3988273e+00 	 1.9801807e-01


.. parsed-literal::

     145	 1.5391331e+00	 1.3309154e-01	 1.5941322e+00	 1.3379496e-01	  1.4005811e+00 	 2.1568871e-01


.. parsed-literal::

     146	 1.5404082e+00	 1.3292795e-01	 1.5954755e+00	 1.3388865e-01	  1.3996942e+00 	 2.1630859e-01
     147	 1.5417431e+00	 1.3272043e-01	 1.5968509e+00	 1.3382200e-01	  1.3988802e+00 	 2.0120502e-01


.. parsed-literal::

     148	 1.5428703e+00	 1.3265913e-01	 1.5979809e+00	 1.3383658e-01	  1.3970491e+00 	 2.0275092e-01


.. parsed-literal::

     149	 1.5444605e+00	 1.3244440e-01	 1.5996093e+00	 1.3373486e-01	  1.3929107e+00 	 2.0650911e-01
     150	 1.5454064e+00	 1.3209610e-01	 1.6006498e+00	 1.3357331e-01	  1.3860615e+00 	 1.9945502e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 3s, sys: 1.11 s, total: 2min 4s
    Wall time: 31.4 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7f6634e24e20>



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
    Process 0 running estimator on chunk 10,000 - 20,000
    Process 0 estimating GPz PZ PDF for rows 10,000 - 20,000


.. parsed-literal::

    Process 0 running estimator on chunk 20,000 - 20,449
    Process 0 estimating GPz PZ PDF for rows 20,000 - 20,449
    CPU times: user 2.07 s, sys: 53.9 ms, total: 2.12 s
    Wall time: 645 ms


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

