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
       1	-3.4989168e-01	 3.2231045e-01	-3.4026306e-01	 3.1362867e-01	[-3.2444540e-01]	 4.6325660e-01


.. parsed-literal::

       2	-2.7813984e-01	 3.1176839e-01	-2.5457741e-01	 3.0215461e-01	[-2.2626151e-01]	 2.2992802e-01


.. parsed-literal::

       3	-2.2803822e-01	 2.8789719e-01	-1.8359733e-01	 2.7763055e-01	[-1.4237248e-01]	 2.7997327e-01


.. parsed-literal::

       4	-1.8960869e-01	 2.7079863e-01	-1.4127647e-01	 2.6324924e-01	[-1.0245322e-01]	 2.8702331e-01
       5	-1.2037830e-01	 2.5858051e-01	-9.0162386e-02	 2.4957965e-01	[-4.3263443e-02]	 2.0071268e-01


.. parsed-literal::

       6	-7.7897412e-02	 2.5421984e-01	-5.0925874e-02	 2.4521101e-01	[-1.8720778e-02]	 1.8479300e-01


.. parsed-literal::

       7	-5.7560020e-02	 2.5043522e-01	-3.5164049e-02	 2.4132652e-01	[-5.7215768e-04]	 2.1617913e-01


.. parsed-literal::

       8	-4.6485090e-02	 2.4851260e-01	-2.7104185e-02	 2.3928116e-01	[ 1.1216769e-02]	 2.1465540e-01


.. parsed-literal::

       9	-3.5093716e-02	 2.4618137e-01	-1.7159567e-02	 2.3786293e-01	[ 1.7182661e-02]	 2.1317363e-01
      10	-2.4462203e-02	 2.4418929e-01	-8.6131278e-03	 2.3621518e-01	[ 2.6893715e-02]	 1.9327164e-01


.. parsed-literal::

      11	-2.0423743e-02	 2.4341639e-01	-5.2016800e-03	 2.3581461e-01	[ 2.8534744e-02]	 1.7985106e-01


.. parsed-literal::

      12	-1.5340107e-02	 2.4252193e-01	-8.3664988e-04	 2.3571376e-01	[ 2.9471019e-02]	 2.1327949e-01


.. parsed-literal::

      13	-1.0090854e-02	 2.4168371e-01	 3.8355706e-03	 2.3502528e-01	[ 3.2355618e-02]	 2.1403170e-01


.. parsed-literal::

      14	 3.1718331e-02	 2.3421201e-01	 4.7688861e-02	 2.2614697e-01	[ 7.9653061e-02]	 3.1814909e-01


.. parsed-literal::

      15	 7.4384974e-02	 2.2987341e-01	 9.6984956e-02	 2.2212952e-01	[ 1.1503316e-01]	 3.3224010e-01


.. parsed-literal::

      16	 1.4572620e-01	 2.2346989e-01	 1.6862457e-01	 2.1603861e-01	[ 1.8885283e-01]	 2.0603204e-01
      17	 2.3800223e-01	 2.1865519e-01	 2.6687283e-01	 2.1423675e-01	[ 3.0377216e-01]	 1.9908309e-01


.. parsed-literal::

      18	 2.7681939e-01	 2.1530894e-01	 3.0616871e-01	 2.0809573e-01	[ 3.4904296e-01]	 2.1043921e-01


.. parsed-literal::

      19	 3.3342814e-01	 2.0962626e-01	 3.6465825e-01	 2.0325088e-01	[ 4.0654923e-01]	 2.0152020e-01


.. parsed-literal::

      20	 4.2178132e-01	 2.1025454e-01	 4.5435145e-01	 2.0272404e-01	[ 5.0240209e-01]	 2.1753716e-01


.. parsed-literal::

      21	 5.2306000e-01	 2.0668018e-01	 5.5862659e-01	 2.0011105e-01	[ 6.0656719e-01]	 2.0774555e-01


.. parsed-literal::

      22	 5.8222693e-01	 2.0258317e-01	 6.2130276e-01	 1.9662725e-01	[ 6.8413273e-01]	 2.2114658e-01


.. parsed-literal::

      23	 6.1514843e-01	 1.9858321e-01	 6.5358416e-01	 1.9390482e-01	[ 7.0672479e-01]	 2.2446775e-01


.. parsed-literal::

      24	 6.4539320e-01	 1.9548177e-01	 6.8360161e-01	 1.8988405e-01	[ 7.3602980e-01]	 2.0879149e-01
      25	 6.9121330e-01	 2.0104602e-01	 7.2844994e-01	 1.9190227e-01	[ 7.7357984e-01]	 1.9573855e-01


.. parsed-literal::

      26	 7.3025706e-01	 1.9965441e-01	 7.6733439e-01	 1.8964129e-01	[ 8.0483367e-01]	 2.0365930e-01


.. parsed-literal::

      27	 7.6572237e-01	 1.9775259e-01	 8.0401323e-01	 1.8921781e-01	[ 8.3644239e-01]	 2.1867514e-01


.. parsed-literal::

      28	 7.9143515e-01	 1.9860814e-01	 8.3053173e-01	 1.8951278e-01	[ 8.5941437e-01]	 2.1053910e-01


.. parsed-literal::

      29	 8.3345205e-01	 1.9700095e-01	 8.7445148e-01	 1.8731193e-01	[ 8.9336877e-01]	 2.1862531e-01


.. parsed-literal::

      30	 8.6740586e-01	 1.9057605e-01	 9.0871848e-01	 1.8162518e-01	[ 9.2472927e-01]	 2.0939565e-01
      31	 8.8589827e-01	 1.8951294e-01	 9.2672609e-01	 1.8052521e-01	[ 9.3740789e-01]	 2.0295787e-01


.. parsed-literal::

      32	 9.0869603e-01	 1.8656225e-01	 9.5073957e-01	 1.7894346e-01	[ 9.6012101e-01]	 1.9883060e-01
      33	 9.3042099e-01	 1.8500500e-01	 9.7318640e-01	 1.7665807e-01	[ 9.7899027e-01]	 1.8865013e-01


.. parsed-literal::

      34	 9.5316941e-01	 1.8206499e-01	 9.9629307e-01	 1.7472401e-01	[ 9.9472874e-01]	 1.8724489e-01


.. parsed-literal::

      35	 9.7337513e-01	 1.7951707e-01	 1.0169513e+00	 1.7229229e-01	[ 1.0130185e+00]	 2.0806742e-01


.. parsed-literal::

      36	 9.8719863e-01	 1.7693680e-01	 1.0319125e+00	 1.6819756e-01	[ 1.0375494e+00]	 2.1030140e-01


.. parsed-literal::

      37	 1.0045616e+00	 1.7465674e-01	 1.0492241e+00	 1.6643974e-01	[ 1.0560122e+00]	 2.1658516e-01


.. parsed-literal::

      38	 1.0132454e+00	 1.7334377e-01	 1.0578918e+00	 1.6515242e-01	[ 1.0652881e+00]	 2.1850610e-01


.. parsed-literal::

      39	 1.0303785e+00	 1.7006259e-01	 1.0757809e+00	 1.6232993e-01	[ 1.0846655e+00]	 2.1477437e-01


.. parsed-literal::

      40	 1.0387364e+00	 1.6999114e-01	 1.0848451e+00	 1.6085534e-01	[ 1.0936858e+00]	 2.2024655e-01


.. parsed-literal::

      41	 1.0522273e+00	 1.6865717e-01	 1.0980693e+00	 1.6003246e-01	[ 1.1060817e+00]	 2.1390224e-01


.. parsed-literal::

      42	 1.0601187e+00	 1.6792926e-01	 1.1061234e+00	 1.5990777e-01	[ 1.1111207e+00]	 2.1256328e-01
      43	 1.0683718e+00	 1.6781802e-01	 1.1145525e+00	 1.6020275e-01	[ 1.1158223e+00]	 1.7909217e-01


.. parsed-literal::

      44	 1.0826233e+00	 1.6710719e-01	 1.1294169e+00	 1.6003966e-01	[ 1.1246346e+00]	 2.0858145e-01


.. parsed-literal::

      45	 1.0953009e+00	 1.6588718e-01	 1.1423083e+00	 1.5822588e-01	[ 1.1329857e+00]	 2.0441699e-01


.. parsed-literal::

      46	 1.1069233e+00	 1.6418838e-01	 1.1538811e+00	 1.5602127e-01	[ 1.1401342e+00]	 2.1553016e-01


.. parsed-literal::

      47	 1.1187203e+00	 1.6115255e-01	 1.1660955e+00	 1.5298327e-01	[ 1.1491316e+00]	 2.2153759e-01


.. parsed-literal::

      48	 1.1290978e+00	 1.5952071e-01	 1.1763853e+00	 1.5130990e-01	[ 1.1565692e+00]	 2.1173525e-01


.. parsed-literal::

      49	 1.1359193e+00	 1.5904022e-01	 1.1831256e+00	 1.5078957e-01	[ 1.1619373e+00]	 2.2092915e-01
      50	 1.1504082e+00	 1.5802959e-01	 1.1980860e+00	 1.4976611e-01	[ 1.1721631e+00]	 1.8068147e-01


.. parsed-literal::

      51	 1.1592759e+00	 1.5710555e-01	 1.2073733e+00	 1.4809238e-01	[ 1.1729486e+00]	 1.9607091e-01


.. parsed-literal::

      52	 1.1689651e+00	 1.5668446e-01	 1.2168733e+00	 1.4753125e-01	[ 1.1823584e+00]	 2.0841479e-01
      53	 1.1779334e+00	 1.5561073e-01	 1.2258446e+00	 1.4600863e-01	[ 1.1883914e+00]	 1.7274761e-01


.. parsed-literal::

      54	 1.1863686e+00	 1.5451711e-01	 1.2345214e+00	 1.4460176e-01	[ 1.1957712e+00]	 2.1530533e-01


.. parsed-literal::

      55	 1.1942663e+00	 1.5297356e-01	 1.2429622e+00	 1.4208709e-01	  1.1950278e+00 	 2.1952486e-01


.. parsed-literal::

      56	 1.2027226e+00	 1.5248601e-01	 1.2513999e+00	 1.4137882e-01	[ 1.2077656e+00]	 2.0314670e-01


.. parsed-literal::

      57	 1.2083753e+00	 1.5213215e-01	 1.2571768e+00	 1.4074703e-01	[ 1.2152856e+00]	 2.0684814e-01


.. parsed-literal::

      58	 1.2177921e+00	 1.5136421e-01	 1.2669344e+00	 1.3963393e-01	[ 1.2258786e+00]	 2.0536971e-01


.. parsed-literal::

      59	 1.2299243e+00	 1.5015783e-01	 1.2797510e+00	 1.3784176e-01	[ 1.2350863e+00]	 2.1017575e-01


.. parsed-literal::

      60	 1.2360408e+00	 1.4966632e-01	 1.2861569e+00	 1.3717321e-01	[ 1.2424593e+00]	 3.2528591e-01
      61	 1.2412352e+00	 1.4908452e-01	 1.2912573e+00	 1.3694305e-01	[ 1.2448088e+00]	 1.9838214e-01


.. parsed-literal::

      62	 1.2490878e+00	 1.4790046e-01	 1.2991169e+00	 1.3621096e-01	[ 1.2525834e+00]	 2.0267558e-01


.. parsed-literal::

      63	 1.2558605e+00	 1.4690188e-01	 1.3060022e+00	 1.3606559e-01	  1.2513178e+00 	 2.1270847e-01


.. parsed-literal::

      64	 1.2636807e+00	 1.4602081e-01	 1.3138704e+00	 1.3546010e-01	[ 1.2588754e+00]	 2.0734191e-01


.. parsed-literal::

      65	 1.2713167e+00	 1.4536197e-01	 1.3216217e+00	 1.3519157e-01	[ 1.2651048e+00]	 2.0722890e-01


.. parsed-literal::

      66	 1.2795585e+00	 1.4470620e-01	 1.3301947e+00	 1.3494682e-01	[ 1.2688392e+00]	 2.0865798e-01


.. parsed-literal::

      67	 1.2892020e+00	 1.4367345e-01	 1.3399648e+00	 1.3456617e-01	[ 1.2762623e+00]	 2.0662308e-01
      68	 1.2984635e+00	 1.4275315e-01	 1.3494414e+00	 1.3385008e-01	[ 1.2802382e+00]	 1.8628526e-01


.. parsed-literal::

      69	 1.3031159e+00	 1.4248817e-01	 1.3540697e+00	 1.3339879e-01	[ 1.2844489e+00]	 2.1495032e-01


.. parsed-literal::

      70	 1.3101688e+00	 1.4158716e-01	 1.3615181e+00	 1.3264198e-01	[ 1.2906863e+00]	 2.0651579e-01
      71	 1.3165073e+00	 1.4152207e-01	 1.3676057e+00	 1.3239652e-01	[ 1.2971020e+00]	 1.7478490e-01


.. parsed-literal::

      72	 1.3230406e+00	 1.4102430e-01	 1.3741165e+00	 1.3211347e-01	[ 1.3049406e+00]	 2.1311569e-01
      73	 1.3311285e+00	 1.4026653e-01	 1.3826585e+00	 1.3189329e-01	[ 1.3111650e+00]	 2.0293665e-01


.. parsed-literal::

      74	 1.3352232e+00	 1.3980228e-01	 1.3868880e+00	 1.3158875e-01	[ 1.3174403e+00]	 2.1779513e-01
      75	 1.3396319e+00	 1.3956211e-01	 1.3913026e+00	 1.3143093e-01	[ 1.3218160e+00]	 1.8098354e-01


.. parsed-literal::

      76	 1.3451268e+00	 1.3898448e-01	 1.3969301e+00	 1.3112919e-01	[ 1.3272075e+00]	 2.1247244e-01


.. parsed-literal::

      77	 1.3492062e+00	 1.3849641e-01	 1.4010226e+00	 1.3110670e-01	[ 1.3311401e+00]	 2.0441747e-01


.. parsed-literal::

      78	 1.3562737e+00	 1.3750460e-01	 1.4080302e+00	 1.3172163e-01	[ 1.3378426e+00]	 2.1261191e-01


.. parsed-literal::

      79	 1.3622450e+00	 1.3741021e-01	 1.4141122e+00	 1.3245479e-01	[ 1.3446122e+00]	 2.1660423e-01


.. parsed-literal::

      80	 1.3662976e+00	 1.3724829e-01	 1.4181640e+00	 1.3224512e-01	[ 1.3468581e+00]	 2.0961046e-01


.. parsed-literal::

      81	 1.3725844e+00	 1.3740558e-01	 1.4250170e+00	 1.3249097e-01	  1.3466524e+00 	 2.0864201e-01
      82	 1.3771144e+00	 1.3700237e-01	 1.4295663e+00	 1.3321474e-01	  1.3466376e+00 	 1.8906617e-01


.. parsed-literal::

      83	 1.3807491e+00	 1.3683573e-01	 1.4330978e+00	 1.3324316e-01	[ 1.3522931e+00]	 1.9239020e-01


.. parsed-literal::

      84	 1.3849547e+00	 1.3661874e-01	 1.4373376e+00	 1.3304889e-01	[ 1.3579615e+00]	 2.0788622e-01


.. parsed-literal::

      85	 1.3886473e+00	 1.3617748e-01	 1.4411008e+00	 1.3280109e-01	[ 1.3608565e+00]	 2.1716404e-01


.. parsed-literal::

      86	 1.3930033e+00	 1.3585035e-01	 1.4454803e+00	 1.3231990e-01	[ 1.3631899e+00]	 2.1224809e-01
      87	 1.3990446e+00	 1.3542741e-01	 1.4517469e+00	 1.3147895e-01	[ 1.3642401e+00]	 1.9026971e-01


.. parsed-literal::

      88	 1.4024020e+00	 1.3508134e-01	 1.4550324e+00	 1.3139377e-01	[ 1.3652414e+00]	 2.1482778e-01


.. parsed-literal::

      89	 1.4067013e+00	 1.3500228e-01	 1.4591648e+00	 1.3137151e-01	[ 1.3716450e+00]	 2.0379257e-01


.. parsed-literal::

      90	 1.4121054e+00	 1.3478153e-01	 1.4645480e+00	 1.3146073e-01	[ 1.3769268e+00]	 2.1999431e-01


.. parsed-literal::

      91	 1.4160460e+00	 1.3442360e-01	 1.4686003e+00	 1.3152336e-01	[ 1.3813304e+00]	 2.0574522e-01


.. parsed-literal::

      92	 1.4204406e+00	 1.3397267e-01	 1.4730731e+00	 1.3151243e-01	[ 1.3842736e+00]	 2.1966743e-01
      93	 1.4236159e+00	 1.3357034e-01	 1.4763198e+00	 1.3099529e-01	[ 1.3843738e+00]	 2.0224309e-01


.. parsed-literal::

      94	 1.4261885e+00	 1.3333841e-01	 1.4789943e+00	 1.3062618e-01	[ 1.3858218e+00]	 2.0801616e-01


.. parsed-literal::

      95	 1.4304097e+00	 1.3297989e-01	 1.4834332e+00	 1.3017870e-01	  1.3849387e+00 	 2.1847296e-01


.. parsed-literal::

      96	 1.4338752e+00	 1.3269268e-01	 1.4872416e+00	 1.3021338e-01	  1.3826233e+00 	 2.1024370e-01
      97	 1.4369981e+00	 1.3259688e-01	 1.4902942e+00	 1.3024312e-01	  1.3853321e+00 	 2.0089221e-01


.. parsed-literal::

      98	 1.4392158e+00	 1.3245518e-01	 1.4924792e+00	 1.3043077e-01	  1.3853419e+00 	 2.2079802e-01


.. parsed-literal::

      99	 1.4415042e+00	 1.3222662e-01	 1.4948367e+00	 1.3051768e-01	  1.3857215e+00 	 2.1824002e-01
     100	 1.4452477e+00	 1.3167310e-01	 1.4987493e+00	 1.3090708e-01	  1.3857341e+00 	 1.9921064e-01


.. parsed-literal::

     101	 1.4486184e+00	 1.3132413e-01	 1.5021489e+00	 1.3072002e-01	[ 1.3871811e+00]	 2.0167851e-01
     102	 1.4523522e+00	 1.3086786e-01	 1.5059326e+00	 1.3018575e-01	[ 1.3898559e+00]	 1.9365382e-01


.. parsed-literal::

     103	 1.4542008e+00	 1.3036584e-01	 1.5078933e+00	 1.2944623e-01	  1.3864740e+00 	 2.1203661e-01
     104	 1.4570800e+00	 1.3024504e-01	 1.5106528e+00	 1.2904381e-01	[ 1.3907015e+00]	 1.9379950e-01


.. parsed-literal::

     105	 1.4589691e+00	 1.3009865e-01	 1.5125238e+00	 1.2871200e-01	[ 1.3915984e+00]	 2.1274400e-01


.. parsed-literal::

     106	 1.4611661e+00	 1.2993709e-01	 1.5146863e+00	 1.2828437e-01	[ 1.3925309e+00]	 2.1135712e-01


.. parsed-literal::

     107	 1.4644955e+00	 1.2962702e-01	 1.5180249e+00	 1.2733373e-01	  1.3921840e+00 	 2.1983242e-01


.. parsed-literal::

     108	 1.4675778e+00	 1.2938554e-01	 1.5210720e+00	 1.2703003e-01	[ 1.3957413e+00]	 2.1438408e-01


.. parsed-literal::

     109	 1.4696566e+00	 1.2930328e-01	 1.5231371e+00	 1.2701838e-01	[ 1.3966177e+00]	 2.1773887e-01


.. parsed-literal::

     110	 1.4720925e+00	 1.2915921e-01	 1.5256412e+00	 1.2717546e-01	  1.3963578e+00 	 2.1585989e-01


.. parsed-literal::

     111	 1.4743103e+00	 1.2901517e-01	 1.5279309e+00	 1.2724568e-01	  1.3959797e+00 	 2.1156526e-01


.. parsed-literal::

     112	 1.4768099e+00	 1.2869975e-01	 1.5306657e+00	 1.2775470e-01	  1.3925558e+00 	 2.1699929e-01


.. parsed-literal::

     113	 1.4807148e+00	 1.2843970e-01	 1.5345342e+00	 1.2729103e-01	  1.3954423e+00 	 2.1851444e-01


.. parsed-literal::

     114	 1.4820174e+00	 1.2834521e-01	 1.5357596e+00	 1.2696321e-01	[ 1.3977536e+00]	 2.1416783e-01
     115	 1.4843447e+00	 1.2811876e-01	 1.5381095e+00	 1.2622544e-01	  1.3968620e+00 	 2.0288134e-01


.. parsed-literal::

     116	 1.4862181e+00	 1.2795827e-01	 1.5401647e+00	 1.2600637e-01	  1.3962188e+00 	 1.9191098e-01
     117	 1.4883491e+00	 1.2787791e-01	 1.5422759e+00	 1.2585401e-01	  1.3946448e+00 	 1.8309379e-01


.. parsed-literal::

     118	 1.4900914e+00	 1.2779773e-01	 1.5440959e+00	 1.2582203e-01	  1.3908856e+00 	 2.1473718e-01


.. parsed-literal::

     119	 1.4914066e+00	 1.2774884e-01	 1.5454356e+00	 1.2582433e-01	  1.3894981e+00 	 2.1072483e-01
     120	 1.4947971e+00	 1.2761198e-01	 1.5489173e+00	 1.2646544e-01	  1.3842998e+00 	 1.9193196e-01


.. parsed-literal::

     121	 1.4966189e+00	 1.2752085e-01	 1.5507198e+00	 1.2607452e-01	  1.3781723e+00 	 3.1156373e-01
     122	 1.4983120e+00	 1.2742449e-01	 1.5523690e+00	 1.2617647e-01	  1.3796621e+00 	 1.8299079e-01


.. parsed-literal::

     123	 1.5003420e+00	 1.2728422e-01	 1.5543791e+00	 1.2618247e-01	  1.3786926e+00 	 2.2147870e-01


.. parsed-literal::

     124	 1.5020561e+00	 1.2709733e-01	 1.5561791e+00	 1.2590956e-01	  1.3764675e+00 	 2.1144271e-01
     125	 1.5049289e+00	 1.2681624e-01	 1.5592045e+00	 1.2530981e-01	  1.3692548e+00 	 1.8104863e-01


.. parsed-literal::

     126	 1.5066806e+00	 1.2660562e-01	 1.5611372e+00	 1.2427478e-01	  1.3614850e+00 	 2.1636033e-01


.. parsed-literal::

     127	 1.5088803e+00	 1.2657965e-01	 1.5632576e+00	 1.2424553e-01	  1.3636866e+00 	 2.0305991e-01


.. parsed-literal::

     128	 1.5102944e+00	 1.2655098e-01	 1.5646371e+00	 1.2423969e-01	  1.3650283e+00 	 2.0649457e-01


.. parsed-literal::

     129	 1.5119382e+00	 1.2643698e-01	 1.5662913e+00	 1.2427993e-01	  1.3633052e+00 	 2.0597386e-01


.. parsed-literal::

     130	 1.5145936e+00	 1.2625372e-01	 1.5690207e+00	 1.2484909e-01	  1.3639737e+00 	 2.0931959e-01


.. parsed-literal::

     131	 1.5166018e+00	 1.2602103e-01	 1.5710690e+00	 1.2540025e-01	  1.3536698e+00 	 2.1513438e-01


.. parsed-literal::

     132	 1.5182389e+00	 1.2599869e-01	 1.5726500e+00	 1.2531737e-01	  1.3570266e+00 	 2.1259093e-01


.. parsed-literal::

     133	 1.5199408e+00	 1.2592013e-01	 1.5743968e+00	 1.2548526e-01	  1.3590065e+00 	 2.0869493e-01
     134	 1.5217021e+00	 1.2578886e-01	 1.5762189e+00	 1.2547788e-01	  1.3585488e+00 	 1.8708038e-01


.. parsed-literal::

     135	 1.5237842e+00	 1.2559810e-01	 1.5784145e+00	 1.2585433e-01	  1.3619913e+00 	 2.0441532e-01


.. parsed-literal::

     136	 1.5256785e+00	 1.2543721e-01	 1.5802855e+00	 1.2548844e-01	  1.3610028e+00 	 2.1101117e-01
     137	 1.5272544e+00	 1.2531674e-01	 1.5818491e+00	 1.2531794e-01	  1.3601303e+00 	 2.0221496e-01


.. parsed-literal::

     138	 1.5290577e+00	 1.2515809e-01	 1.5836931e+00	 1.2534200e-01	  1.3611105e+00 	 2.1077204e-01


.. parsed-literal::

     139	 1.5305166e+00	 1.2491714e-01	 1.5853870e+00	 1.2588614e-01	  1.3509326e+00 	 2.0896769e-01


.. parsed-literal::

     140	 1.5332099e+00	 1.2485080e-01	 1.5880149e+00	 1.2596202e-01	  1.3589359e+00 	 2.1756268e-01


.. parsed-literal::

     141	 1.5343587e+00	 1.2482018e-01	 1.5891921e+00	 1.2619615e-01	  1.3583960e+00 	 2.1936202e-01
     142	 1.5359982e+00	 1.2476676e-01	 1.5909150e+00	 1.2651816e-01	  1.3547309e+00 	 1.9974399e-01


.. parsed-literal::

     143	 1.5371851e+00	 1.2464239e-01	 1.5922949e+00	 1.2677927e-01	  1.3416201e+00 	 2.0592046e-01


.. parsed-literal::

     144	 1.5390813e+00	 1.2458922e-01	 1.5941837e+00	 1.2666416e-01	  1.3384466e+00 	 2.1903443e-01


.. parsed-literal::

     145	 1.5401855e+00	 1.2452974e-01	 1.5952790e+00	 1.2640195e-01	  1.3377735e+00 	 2.0435143e-01
     146	 1.5415901e+00	 1.2448042e-01	 1.5967139e+00	 1.2620440e-01	  1.3350917e+00 	 1.8869257e-01


.. parsed-literal::

     147	 1.5425251e+00	 1.2448769e-01	 1.5978130e+00	 1.2657787e-01	  1.3233440e+00 	 2.0825386e-01


.. parsed-literal::

     148	 1.5442954e+00	 1.2446733e-01	 1.5995237e+00	 1.2637345e-01	  1.3271331e+00 	 2.0547366e-01


.. parsed-literal::

     149	 1.5452455e+00	 1.2445888e-01	 1.6005088e+00	 1.2661980e-01	  1.3257345e+00 	 2.0195174e-01
     150	 1.5464991e+00	 1.2446236e-01	 1.6018227e+00	 1.2694463e-01	  1.3221353e+00 	 1.9679952e-01


.. parsed-literal::

    Inserting handle into data store.  model_GPz_Train: inprogress_GPz_model.pkl, GPz_Train
    CPU times: user 2min 6s, sys: 1.23 s, total: 2min 7s
    Wall time: 32.1 s




.. parsed-literal::

    <rail.core.data.ModelHandle at 0x7ff288c912d0>



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
    CPU times: user 2.04 s, sys: 64.9 ms, total: 2.11 s
    Wall time: 632 ms


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

