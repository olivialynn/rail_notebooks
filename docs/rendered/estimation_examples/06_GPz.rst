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
       1	-3.3098353e-01	 3.1650498e-01	-3.2113078e-01	 3.3650381e-01	[-3.6145082e-01]	 4.6180868e-01


.. parsed-literal::

       2	-2.5148213e-01	 3.0228048e-01	-2.2423780e-01	 3.2570266e-01	[-2.9946344e-01]	 2.3265243e-01


.. parsed-literal::

       3	-2.1041004e-01	 2.8519137e-01	-1.6956331e-01	 3.1114974e-01	[-2.8505132e-01]	 2.9808593e-01
       4	-1.7233346e-01	 2.5959157e-01	-1.3144927e-01	 2.8599162e-01	 -3.0158441e-01 	 1.8226337e-01


.. parsed-literal::

       5	-8.8371749e-02	 2.5197299e-01	-4.9328809e-02	 2.8112557e-01	[-1.9623423e-01]	 2.0552540e-01


.. parsed-literal::

       6	-5.2455944e-02	 2.4596222e-01	-1.7207075e-02	 2.7368011e-01	[-1.2879374e-01]	 2.0252776e-01


.. parsed-literal::

       7	-3.3004036e-02	 2.4328373e-01	-5.9284663e-03	 2.7064420e-01	[-1.2309325e-01]	 2.0399547e-01


.. parsed-literal::

       8	-1.9416060e-02	 2.4116746e-01	 3.0533345e-03	 2.6879001e-01	[-1.1632267e-01]	 2.1741366e-01


.. parsed-literal::

       9	-3.7813082e-03	 2.3844694e-01	 1.4764561e-02	 2.6773145e-01	[-1.1433007e-01]	 2.0947814e-01


.. parsed-literal::

      10	 4.7794157e-03	 2.3723143e-01	 2.0669408e-02	 2.6653468e-01	[-1.0432023e-01]	 2.4064469e-01


.. parsed-literal::

      11	 1.1755216e-02	 2.3583614e-01	 2.6513875e-02	 2.6550526e-01	 -1.0721660e-01 	 2.0719910e-01


.. parsed-literal::

      12	 1.4091143e-02	 2.3542476e-01	 2.8587060e-02	 2.6500977e-01	[-1.0212959e-01]	 2.1357679e-01


.. parsed-literal::

      13	 1.8583829e-02	 2.3464334e-01	 3.2634923e-02	 2.6369969e-01	[-9.9114339e-02]	 2.1375918e-01


.. parsed-literal::

      14	 2.3842417e-02	 2.3343920e-01	 3.8137573e-02	 2.6208877e-01	[-8.0802035e-02]	 2.0468998e-01


.. parsed-literal::

      15	 8.5475574e-02	 2.2269543e-01	 1.0346262e-01	 2.5007457e-01	[ 8.5304408e-03]	 3.2467580e-01


.. parsed-literal::

      16	 1.0725862e-01	 2.2027095e-01	 1.2765074e-01	 2.4464098e-01	[ 5.6689179e-02]	 3.0132651e-01


.. parsed-literal::

      17	 1.6535769e-01	 2.1597711e-01	 1.8604331e-01	 2.4240490e-01	[ 1.0714597e-01]	 2.0370173e-01


.. parsed-literal::

      18	 2.7216090e-01	 2.0996177e-01	 3.0072100e-01	 2.3547106e-01	[ 1.9804191e-01]	 2.0961809e-01


.. parsed-literal::

      19	 3.2506496e-01	 2.0372406e-01	 3.5613953e-01	 2.2907689e-01	[ 2.4404643e-01]	 2.1754837e-01


.. parsed-literal::

      20	 3.8040470e-01	 1.9768112e-01	 4.1176623e-01	 2.2289487e-01	[ 2.9421900e-01]	 2.1278381e-01


.. parsed-literal::

      21	 4.5172234e-01	 1.9249536e-01	 4.8370264e-01	 2.1781941e-01	[ 3.7381198e-01]	 2.1591640e-01


.. parsed-literal::

      22	 5.4832551e-01	 1.8805706e-01	 5.8133699e-01	 2.1146574e-01	[ 4.8764369e-01]	 2.1124816e-01


.. parsed-literal::

      23	 6.4201920e-01	 1.8708157e-01	 6.7818762e-01	 2.1287557e-01	[ 5.7263078e-01]	 2.1927047e-01


.. parsed-literal::

      24	 6.6391235e-01	 1.8844332e-01	 7.0291294e-01	 2.1670366e-01	[ 5.7311995e-01]	 2.2090101e-01
      25	 7.0458752e-01	 1.8298539e-01	 7.4104091e-01	 2.1100829e-01	[ 6.4424808e-01]	 1.9410181e-01


.. parsed-literal::

      26	 7.2691375e-01	 1.8296820e-01	 7.6340968e-01	 2.1037470e-01	[ 6.6058377e-01]	 1.8849158e-01


.. parsed-literal::

      27	 7.5618604e-01	 1.8101152e-01	 7.9325521e-01	 2.0699772e-01	[ 6.9528602e-01]	 2.1431589e-01


.. parsed-literal::

      28	 7.8399270e-01	 1.8034427e-01	 8.2198180e-01	 2.0822040e-01	[ 7.2093476e-01]	 2.1178269e-01


.. parsed-literal::

      29	 8.0619723e-01	 1.7769297e-01	 8.4441133e-01	 2.0567208e-01	[ 7.5068540e-01]	 2.0574212e-01


.. parsed-literal::

      30	 8.2916801e-01	 1.7580333e-01	 8.6773167e-01	 2.0359050e-01	[ 7.8256770e-01]	 2.1673465e-01


.. parsed-literal::

      31	 8.5036552e-01	 1.7458608e-01	 8.8944177e-01	 2.0317552e-01	[ 8.1510059e-01]	 2.2053337e-01
      32	 8.6688674e-01	 1.7379067e-01	 9.0640440e-01	 2.0131115e-01	[ 8.3880438e-01]	 1.9521928e-01


.. parsed-literal::

      33	 8.8294538e-01	 1.7304512e-01	 9.2281183e-01	 2.0092782e-01	[ 8.5630849e-01]	 2.0597053e-01


.. parsed-literal::

      34	 9.0957522e-01	 1.7363808e-01	 9.5041061e-01	 2.0593994e-01	[ 8.7352868e-01]	 2.1073318e-01


.. parsed-literal::

      35	 9.3118345e-01	 1.7347450e-01	 9.7246042e-01	 2.0710878e-01	[ 8.9668878e-01]	 2.0423293e-01


.. parsed-literal::

      36	 9.4531067e-01	 1.7260577e-01	 9.8671632e-01	 2.0582176e-01	[ 9.0734623e-01]	 2.1400499e-01


.. parsed-literal::

      37	 9.5685444e-01	 1.7238871e-01	 9.9845846e-01	 2.0559211e-01	[ 9.1648212e-01]	 2.0563269e-01
      38	 9.7184714e-01	 1.7175615e-01	 1.0138832e+00	 2.0442266e-01	[ 9.3186719e-01]	 1.8287325e-01


.. parsed-literal::

      39	 9.8778156e-01	 1.7076180e-01	 1.0306004e+00	 2.0271095e-01	[ 9.5112096e-01]	 1.8506026e-01


.. parsed-literal::

      40	 1.0027445e+00	 1.6925764e-01	 1.0457583e+00	 2.0060513e-01	[ 9.7022817e-01]	 2.0653296e-01


.. parsed-literal::

      41	 1.0173024e+00	 1.6721164e-01	 1.0610293e+00	 1.9905471e-01	[ 9.8324178e-01]	 2.0829105e-01


.. parsed-literal::

      42	 1.0280283e+00	 1.6562383e-01	 1.0723827e+00	 1.9905481e-01	[ 9.9003925e-01]	 2.0540214e-01


.. parsed-literal::

      43	 1.0408747e+00	 1.6444048e-01	 1.0859026e+00	 1.9931331e-01	[ 9.9989708e-01]	 2.2006321e-01
      44	 1.0500351e+00	 1.6433905e-01	 1.0954980e+00	 2.0123982e-01	[ 1.0009984e+00]	 1.9506288e-01


.. parsed-literal::

      45	 1.0584563e+00	 1.6393803e-01	 1.1037628e+00	 2.0020574e-01	[ 1.0131961e+00]	 2.1107769e-01


.. parsed-literal::

      46	 1.0667280e+00	 1.6351975e-01	 1.1122081e+00	 1.9914649e-01	[ 1.0226773e+00]	 2.1160507e-01


.. parsed-literal::

      47	 1.0768379e+00	 1.6270200e-01	 1.1224679e+00	 1.9820472e-01	[ 1.0315186e+00]	 2.1645236e-01
      48	 1.0889482e+00	 1.6015838e-01	 1.1352427e+00	 1.9496590e-01	[ 1.0402315e+00]	 1.9081473e-01


.. parsed-literal::

      49	 1.0987938e+00	 1.5823047e-01	 1.1452312e+00	 1.9397307e-01	[ 1.0470268e+00]	 1.8715954e-01


.. parsed-literal::

      50	 1.1058779e+00	 1.5683644e-01	 1.1520422e+00	 1.9183030e-01	[ 1.0554557e+00]	 2.1218586e-01
      51	 1.1129884e+00	 1.5515210e-01	 1.1593134e+00	 1.9007221e-01	[ 1.0637793e+00]	 1.8155980e-01


.. parsed-literal::

      52	 1.1202958e+00	 1.5396314e-01	 1.1666921e+00	 1.8930649e-01	[ 1.0684074e+00]	 1.8694305e-01
      53	 1.1302782e+00	 1.5194212e-01	 1.1772239e+00	 1.8690220e-01	[ 1.0770667e+00]	 1.8910933e-01


.. parsed-literal::

      54	 1.1393290e+00	 1.5151165e-01	 1.1863375e+00	 1.8625447e-01	[ 1.0826706e+00]	 2.0486474e-01
      55	 1.1458307e+00	 1.5103833e-01	 1.1928014e+00	 1.8483374e-01	[ 1.0900886e+00]	 2.0007420e-01


.. parsed-literal::

      56	 1.1552579e+00	 1.4995735e-01	 1.2027663e+00	 1.8184886e-01	[ 1.0979526e+00]	 1.9524026e-01
      57	 1.1614896e+00	 1.4930347e-01	 1.2094860e+00	 1.8129933e-01	[ 1.0996207e+00]	 1.8755221e-01


.. parsed-literal::

      58	 1.1692179e+00	 1.4850941e-01	 1.2172005e+00	 1.8008577e-01	[ 1.1070763e+00]	 2.1051645e-01
      59	 1.1791889e+00	 1.4715942e-01	 1.2274928e+00	 1.7804020e-01	[ 1.1154085e+00]	 1.8997502e-01


.. parsed-literal::

      60	 1.1864299e+00	 1.4647724e-01	 1.2349712e+00	 1.7720952e-01	[ 1.1202652e+00]	 1.8633938e-01


.. parsed-literal::

      61	 1.1930473e+00	 1.4572129e-01	 1.2422771e+00	 1.7597027e-01	[ 1.1256503e+00]	 2.0288301e-01


.. parsed-literal::

      62	 1.2007925e+00	 1.4561935e-01	 1.2499180e+00	 1.7637586e-01	[ 1.1305403e+00]	 2.1602631e-01


.. parsed-literal::

      63	 1.2046650e+00	 1.4560968e-01	 1.2536639e+00	 1.7642514e-01	[ 1.1360218e+00]	 2.0488572e-01


.. parsed-literal::

      64	 1.2126415e+00	 1.4505899e-01	 1.2620027e+00	 1.7621258e-01	[ 1.1425347e+00]	 2.0808768e-01


.. parsed-literal::

      65	 1.2202254e+00	 1.4362027e-01	 1.2699066e+00	 1.7453314e-01	[ 1.1562781e+00]	 2.1191835e-01


.. parsed-literal::

      66	 1.2280787e+00	 1.4278802e-01	 1.2777723e+00	 1.7398082e-01	[ 1.1626109e+00]	 2.1052718e-01


.. parsed-literal::

      67	 1.2361372e+00	 1.4130413e-01	 1.2858279e+00	 1.7239090e-01	[ 1.1698567e+00]	 2.1573472e-01
      68	 1.2437463e+00	 1.4009228e-01	 1.2935759e+00	 1.7101939e-01	[ 1.1752470e+00]	 1.9953847e-01


.. parsed-literal::

      69	 1.2509083e+00	 1.3883563e-01	 1.3010398e+00	 1.6868596e-01	[ 1.1825573e+00]	 2.1324110e-01
      70	 1.2595480e+00	 1.3846033e-01	 1.3097749e+00	 1.6853591e-01	[ 1.1859504e+00]	 1.9557095e-01


.. parsed-literal::

      71	 1.2647573e+00	 1.3834442e-01	 1.3148964e+00	 1.6864817e-01	[ 1.1877490e+00]	 1.8853831e-01
      72	 1.2699871e+00	 1.3824768e-01	 1.3203826e+00	 1.6902455e-01	  1.1865244e+00 	 1.9459653e-01


.. parsed-literal::

      73	 1.2744287e+00	 1.3782598e-01	 1.3249425e+00	 1.6821506e-01	[ 1.1883295e+00]	 2.0977092e-01


.. parsed-literal::

      74	 1.2779479e+00	 1.3741113e-01	 1.3284061e+00	 1.6774232e-01	[ 1.1919488e+00]	 2.1355891e-01


.. parsed-literal::

      75	 1.2837598e+00	 1.3657850e-01	 1.3343159e+00	 1.6635177e-01	[ 1.1986012e+00]	 2.0229197e-01


.. parsed-literal::

      76	 1.2898704e+00	 1.3578901e-01	 1.3406695e+00	 1.6515183e-01	[ 1.2044407e+00]	 2.1866512e-01
      77	 1.2961956e+00	 1.3488349e-01	 1.3474002e+00	 1.6264519e-01	[ 1.2084280e+00]	 1.8565369e-01


.. parsed-literal::

      78	 1.3023447e+00	 1.3432243e-01	 1.3536025e+00	 1.6166985e-01	[ 1.2169583e+00]	 2.0099378e-01


.. parsed-literal::

      79	 1.3068638e+00	 1.3424091e-01	 1.3580221e+00	 1.6165814e-01	[ 1.2222800e+00]	 2.1565890e-01


.. parsed-literal::

      80	 1.3129806e+00	 1.3420004e-01	 1.3643576e+00	 1.6106879e-01	[ 1.2247284e+00]	 2.0966554e-01
      81	 1.3176708e+00	 1.3433057e-01	 1.3692150e+00	 1.6108196e-01	[ 1.2282686e+00]	 1.8497539e-01


.. parsed-literal::

      82	 1.3224646e+00	 1.3419448e-01	 1.3739042e+00	 1.6071382e-01	[ 1.2330070e+00]	 2.0429850e-01
      83	 1.3261224e+00	 1.3383910e-01	 1.3775868e+00	 1.6012808e-01	[ 1.2371592e+00]	 1.9660592e-01


.. parsed-literal::

      84	 1.3303500e+00	 1.3329685e-01	 1.3819400e+00	 1.5953577e-01	[ 1.2393339e+00]	 2.1779943e-01


.. parsed-literal::

      85	 1.3356734e+00	 1.3228219e-01	 1.3876426e+00	 1.5751859e-01	[ 1.2455853e+00]	 2.0931196e-01


.. parsed-literal::

      86	 1.3426003e+00	 1.3163034e-01	 1.3945771e+00	 1.5693742e-01	[ 1.2463768e+00]	 2.0327282e-01


.. parsed-literal::

      87	 1.3462088e+00	 1.3130508e-01	 1.3981850e+00	 1.5681634e-01	[ 1.2464491e+00]	 2.1927953e-01


.. parsed-literal::

      88	 1.3502277e+00	 1.3080701e-01	 1.4023310e+00	 1.5621878e-01	[ 1.2466771e+00]	 2.0383120e-01


.. parsed-literal::

      89	 1.3542153e+00	 1.3001185e-01	 1.4065249e+00	 1.5671536e-01	  1.2390589e+00 	 2.1729422e-01


.. parsed-literal::

      90	 1.3597043e+00	 1.2972565e-01	 1.4119112e+00	 1.5614820e-01	  1.2458404e+00 	 2.1958208e-01


.. parsed-literal::

      91	 1.3621839e+00	 1.2955775e-01	 1.4143591e+00	 1.5620955e-01	[ 1.2479356e+00]	 2.1986580e-01


.. parsed-literal::

      92	 1.3662016e+00	 1.2936180e-01	 1.4183513e+00	 1.5670093e-01	[ 1.2515856e+00]	 2.0970297e-01
      93	 1.3692403e+00	 1.2852365e-01	 1.4216670e+00	 1.5737424e-01	  1.2437609e+00 	 1.8827033e-01


.. parsed-literal::

      94	 1.3745242e+00	 1.2798343e-01	 1.4268746e+00	 1.5664472e-01	  1.2514613e+00 	 1.8549037e-01


.. parsed-literal::

      95	 1.3776490e+00	 1.2746552e-01	 1.4300371e+00	 1.5585689e-01	[ 1.2534461e+00]	 2.1294379e-01


.. parsed-literal::

      96	 1.3810939e+00	 1.2672463e-01	 1.4335592e+00	 1.5488604e-01	[ 1.2559528e+00]	 2.1209931e-01
      97	 1.3856934e+00	 1.2622343e-01	 1.4381749e+00	 1.5411832e-01	[ 1.2610653e+00]	 1.9697404e-01


.. parsed-literal::

      98	 1.3897062e+00	 1.2519592e-01	 1.4421775e+00	 1.5314963e-01	[ 1.2732030e+00]	 2.1754742e-01


.. parsed-literal::

      99	 1.3931258e+00	 1.2548945e-01	 1.4454574e+00	 1.5377716e-01	[ 1.2759590e+00]	 2.0477295e-01


.. parsed-literal::

     100	 1.3964646e+00	 1.2561315e-01	 1.4487165e+00	 1.5447674e-01	[ 1.2798517e+00]	 2.1969104e-01


.. parsed-literal::

     101	 1.3996195e+00	 1.2546507e-01	 1.4519245e+00	 1.5454338e-01	[ 1.2817009e+00]	 2.0717788e-01


.. parsed-literal::

     102	 1.4031272e+00	 1.2512714e-01	 1.4554580e+00	 1.5443060e-01	[ 1.2837120e+00]	 2.0169449e-01


.. parsed-literal::

     103	 1.4058689e+00	 1.2447496e-01	 1.4582738e+00	 1.5333693e-01	[ 1.2862233e+00]	 2.2141385e-01


.. parsed-literal::

     104	 1.4089354e+00	 1.2376472e-01	 1.4613922e+00	 1.5194240e-01	[ 1.2878362e+00]	 2.0707393e-01
     105	 1.4135039e+00	 1.2286352e-01	 1.4660241e+00	 1.5004154e-01	[ 1.2917448e+00]	 1.8918037e-01


.. parsed-literal::

     106	 1.4173310e+00	 1.2224713e-01	 1.4699524e+00	 1.4861328e-01	  1.2858710e+00 	 2.1569133e-01


.. parsed-literal::

     107	 1.4204955e+00	 1.2204235e-01	 1.4730953e+00	 1.4875024e-01	  1.2863494e+00 	 2.1718383e-01


.. parsed-literal::

     108	 1.4227364e+00	 1.2205426e-01	 1.4753091e+00	 1.4914371e-01	  1.2886371e+00 	 2.0540357e-01


.. parsed-literal::

     109	 1.4265257e+00	 1.2201915e-01	 1.4792763e+00	 1.4948957e-01	  1.2892555e+00 	 2.1669102e-01


.. parsed-literal::

     110	 1.4295664e+00	 1.2121859e-01	 1.4826466e+00	 1.4853359e-01	  1.2905557e+00 	 2.1458817e-01
     111	 1.4326676e+00	 1.2117600e-01	 1.4857798e+00	 1.4813206e-01	[ 1.2932391e+00]	 1.8407512e-01


.. parsed-literal::

     112	 1.4352438e+00	 1.2084311e-01	 1.4883469e+00	 1.4745814e-01	[ 1.2966990e+00]	 1.8773794e-01


.. parsed-literal::

     113	 1.4385423e+00	 1.2031769e-01	 1.4917222e+00	 1.4677552e-01	[ 1.3003702e+00]	 2.0956826e-01
     114	 1.4412473e+00	 1.1953895e-01	 1.4946313e+00	 1.4594467e-01	[ 1.3110797e+00]	 1.9122982e-01


.. parsed-literal::

     115	 1.4446747e+00	 1.1944913e-01	 1.4979684e+00	 1.4622120e-01	[ 1.3117435e+00]	 1.9176412e-01


.. parsed-literal::

     116	 1.4464176e+00	 1.1951755e-01	 1.4997178e+00	 1.4653287e-01	  1.3114626e+00 	 2.1155214e-01
     117	 1.4489656e+00	 1.1939841e-01	 1.5022561e+00	 1.4664788e-01	[ 1.3123605e+00]	 1.9597912e-01


.. parsed-literal::

     118	 1.4511252e+00	 1.1928437e-01	 1.5044959e+00	 1.4690051e-01	  1.3066788e+00 	 2.0955944e-01


.. parsed-literal::

     119	 1.4543999e+00	 1.1901614e-01	 1.5077094e+00	 1.4673651e-01	  1.3123030e+00 	 2.1150160e-01


.. parsed-literal::

     120	 1.4559613e+00	 1.1878384e-01	 1.5092350e+00	 1.4643912e-01	[ 1.3160229e+00]	 2.0711398e-01


.. parsed-literal::

     121	 1.4592829e+00	 1.1836148e-01	 1.5125165e+00	 1.4623459e-01	[ 1.3231226e+00]	 2.0910215e-01


.. parsed-literal::

     122	 1.4615570e+00	 1.1760828e-01	 1.5148264e+00	 1.4622589e-01	[ 1.3316248e+00]	 2.0721126e-01


.. parsed-literal::

     123	 1.4654152e+00	 1.1752100e-01	 1.5185758e+00	 1.4640016e-01	[ 1.3341698e+00]	 2.1312547e-01


.. parsed-literal::

     124	 1.4675158e+00	 1.1742587e-01	 1.5206410e+00	 1.4669626e-01	[ 1.3356945e+00]	 2.1113110e-01


.. parsed-literal::

     125	 1.4698307e+00	 1.1725337e-01	 1.5229573e+00	 1.4697325e-01	[ 1.3365177e+00]	 2.0951700e-01


.. parsed-literal::

     126	 1.4718759e+00	 1.1686139e-01	 1.5250798e+00	 1.4700608e-01	[ 1.3379443e+00]	 2.0388627e-01


.. parsed-literal::

     127	 1.4744443e+00	 1.1669426e-01	 1.5276343e+00	 1.4675907e-01	[ 1.3396463e+00]	 2.1460152e-01


.. parsed-literal::

     128	 1.4766221e+00	 1.1643533e-01	 1.5298607e+00	 1.4638570e-01	[ 1.3405185e+00]	 2.1361470e-01


.. parsed-literal::

     129	 1.4784429e+00	 1.1625670e-01	 1.5317152e+00	 1.4630057e-01	[ 1.3430240e+00]	 2.1522903e-01


.. parsed-literal::

     130	 1.4802140e+00	 1.1594208e-01	 1.5336023e+00	 1.4636955e-01	  1.3423538e+00 	 2.1152568e-01
     131	 1.4831112e+00	 1.1583156e-01	 1.5364107e+00	 1.4655638e-01	[ 1.3504059e+00]	 2.0016789e-01


.. parsed-literal::

     132	 1.4844496e+00	 1.1580224e-01	 1.5376844e+00	 1.4675351e-01	[ 1.3531599e+00]	 1.9962478e-01


.. parsed-literal::

     133	 1.4863957e+00	 1.1567968e-01	 1.5395742e+00	 1.4686859e-01	[ 1.3571523e+00]	 2.0276237e-01


.. parsed-literal::

     134	 1.4878019e+00	 1.1558937e-01	 1.5409788e+00	 1.4685098e-01	  1.3562629e+00 	 3.2797551e-01


.. parsed-literal::

     135	 1.4893018e+00	 1.1547930e-01	 1.5424885e+00	 1.4667124e-01	  1.3569742e+00 	 2.1151328e-01
     136	 1.4910298e+00	 1.1538151e-01	 1.5442825e+00	 1.4647638e-01	  1.3555396e+00 	 2.0485234e-01


.. parsed-literal::

     137	 1.4927316e+00	 1.1533736e-01	 1.5460472e+00	 1.4642618e-01	  1.3540498e+00 	 2.1925688e-01
     138	 1.4938675e+00	 1.1540723e-01	 1.5473559e+00	 1.4638669e-01	  1.3470038e+00 	 2.0254683e-01


.. parsed-literal::

     139	 1.4959678e+00	 1.1537259e-01	 1.5494097e+00	 1.4643190e-01	  1.3501616e+00 	 2.0842171e-01


.. parsed-literal::

     140	 1.4968443e+00	 1.1535933e-01	 1.5502548e+00	 1.4644511e-01	  1.3518601e+00 	 2.0983887e-01


.. parsed-literal::

     141	 1.4986348e+00	 1.1537684e-01	 1.5520417e+00	 1.4640796e-01	  1.3533964e+00 	 2.2189641e-01


.. parsed-literal::

     142	 1.5011779e+00	 1.1542646e-01	 1.5546327e+00	 1.4632215e-01	  1.3541672e+00 	 2.1981239e-01


.. parsed-literal::

     143	 1.5028699e+00	 1.1550618e-01	 1.5564143e+00	 1.4632326e-01	  1.3527889e+00 	 3.2812715e-01
     144	 1.5050196e+00	 1.1559459e-01	 1.5586438e+00	 1.4629140e-01	  1.3534494e+00 	 1.7934823e-01


.. parsed-literal::

     145	 1.5065368e+00	 1.1562210e-01	 1.5602136e+00	 1.4641054e-01	  1.3504457e+00 	 2.0082378e-01


.. parsed-literal::

     146	 1.5079612e+00	 1.1561543e-01	 1.5616914e+00	 1.4656256e-01	  1.3512293e+00 	 2.1989608e-01


.. parsed-literal::

     147	 1.5094860e+00	 1.1565014e-01	 1.5632722e+00	 1.4688380e-01	  1.3481347e+00 	 2.2002411e-01


.. parsed-literal::

     148	 1.5114711e+00	 1.1562413e-01	 1.5653362e+00	 1.4745277e-01	  1.3473852e+00 	 2.1135306e-01


.. parsed-literal::

     149	 1.5129928e+00	 1.1568741e-01	 1.5669209e+00	 1.4786850e-01	  1.3470364e+00 	 2.1708632e-01


.. parsed-literal::

     150	 1.5141527e+00	 1.1563693e-01	 1.5680452e+00	 1.4787760e-01	  1.3494406e+00 	 2.2166872e-01
    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.31 s, total: 2min 7s
    Wall time: 32 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7fc430921780>



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


.. parsed-literal::

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
    CPU times: user 2.07 s, sys: 58 ms, total: 2.13 s
    Wall time: 642 ms


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

