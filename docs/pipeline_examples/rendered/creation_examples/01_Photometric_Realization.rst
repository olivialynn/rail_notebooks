Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``00_Quick_Start_in_Creation.ipynb``

**Note:** If you’re planning to run this in a notebook, you may want to
use interactive mode instead. See
`Photometric_Realization.ipynb <https://github.com/LSSTDESC/rail/blob/main/interactive_examples/creation_examples/Photometric_Realization.ipynb>`__
in the ``interactive_examples/creation_examples/`` folder for a version
of this notebook in interactive mode.

.. code:: ipython3

    import matplotlib.pyplot as plt
    from pzflow.examples import get_example_flow
    from rail.creation.engines.flowEngine import FlowCreator
    from rail.creation.degraders.photometric_errors import LSSTErrorModel
    from rail.core.stage import RailStage


Specify the path to the pretrained ‘pzflow’ used to generate samples

.. code:: ipython3

    import pzflow
    import os
    
    flow_file = os.path.join(
        os.path.dirname(pzflow.__file__), "example_files", "example-flow.pzflow.pkl"
    )


“True” Engine
~~~~~~~~~~~~~

First, let’s make an Engine that has no degradation. We can use it to
generate a “true” sample, to which we can compare all the degraded
samples below.

Note: in this example, we will use a normalizing flow engine from the
`pzflow <https://github.com/jfcrenshaw/pzflow>`__ package. However,
everything in this notebook is totally agnostic to what the underlying
engine is.

The Engine is a type of RailStage object, so we can make one using the
``RailStage.make_stage`` function for the class of Engine that we want.
We then pass in the configuration parameters as arguments to
``make_stage``.

.. code:: ipython3

    n_samples = int(1e5)
    flowEngine_truth = FlowCreator.make_stage(
        name="truth", model=flow_file, n_samples=n_samples
    )



.. parsed-literal::

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.20/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7efc3b5055a0>



Now we invoke the ``sample`` method to generate some samples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this will return a ``DataHandle`` object, which can keep both
the data itself, and also the path to where the data is written. When
talking to rail stages we can use this as though it were the underlying
data and pass it as an argument. This allows the rail stages to keep
track of where their inputs are coming from.

To calculate magnitude error for extended sources, we need the
information about major and minor axes of each galaxy. Here we simply
generate random values

.. code:: ipython3

    samples_truth = flowEngine_truth.sample(n_samples, seed=0)
    
    import numpy as np
    
    samples_truth.data["major"] = np.abs(
        np.random.normal(loc=0.01, scale=0.1, size=n_samples)
    )  # add major and minor axes
    b_to_a = 1 - 0.5 * np.random.rand(n_samples)
    samples_truth.data["minor"] = samples_truth.data["major"] * b_to_a
    
    print(samples_truth())
    print("Data was written to ", samples_truth.path)



.. parsed-literal::

    Inserting handle into data store.  output_truth: inprogress_output_truth.pq, truth
           redshift          u          g          r          i          z  \
    0      1.398944  27.667536  26.723337  26.032637  25.178587  24.695955   
    1      2.285624  28.786999  27.476589  26.640175  26.259745  25.865673   
    2      1.495132  30.011349  29.789337  28.200390  26.014826  25.030174   
    3      0.842594  29.306244  28.721798  27.353018  26.256907  25.529823   
    4      1.588960  26.273870  26.115387  25.950441  25.687405  25.466606   
    ...         ...        ...        ...        ...        ...        ...   
    99995  0.389450  27.270800  26.371506  25.436853  25.077412  24.852779   
    99996  1.481047  27.478113  26.735254  26.042776  25.204935  24.825092   
    99997  2.023548  26.990147  26.714737  26.377949  26.250343  25.917370   
    99998  1.548204  26.367432  26.206884  26.087980  25.876932  25.715893   
    99999  1.739491  26.881983  26.773064  26.553123  26.319622  25.955982   
    
                   y     major     minor  
    0      23.994413  0.048945  0.044909  
    1      25.391064  0.042682  0.027338  
    2      24.304707  0.141275  0.105956  
    3      25.291103  0.125472  0.121678  
    4      25.096743  0.132289  0.128353  
    ...          ...       ...       ...  
    99995  24.737946  0.076085  0.059943  
    99996  24.224169  0.101686  0.100505  
    99997  25.613836  0.088952  0.083788  
    99998  25.274899  0.017071  0.016200  
    99999  25.699642  0.164908  0.084711  
    
    [100000 rows x 9 columns]
    Data was written to  output_truth.pq


LSSTErrorModel
~~~~~~~~~~~~~~

Now, we will demonstrate the ``LSSTErrorModel``, which adds photometric
errors using a model similar to the model from `Ivezic et
al. 2019 <https://arxiv.org/abs/0805.2366>`__ (specifically, it uses the
model from this paper, without making the high SNR assumption. To
restore this assumption and therefore use the exact model from the
paper, set ``highSNR=True``.)

Let’s create an error model with the default settings for point sources:

.. code:: ipython3

    errorModel = LSSTErrorModel.make_stage(name="error_model")


For extended sources:

.. code:: ipython3

    errorModel_auto = LSSTErrorModel.make_stage(
        name="error_model_auto", extendedType="auto"
    )


.. code:: ipython3

    errorModel_gaap = LSSTErrorModel.make_stage(
        name="error_model_gaap", extendedType="gaap"
    )


Now let’s add this error model as a degrader and draw some samples with
photometric errors.

.. code:: ipython3

    samples_w_errs = errorModel(samples_truth)
    samples_w_errs()



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model
    Inserting handle into data store.  output_error_model: inprogress_output_error_model.pq, error_model




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>30.342253</td>
          <td>2.959663</td>
          <td>26.572858</td>
          <td>0.147449</td>
          <td>26.107050</td>
          <td>0.086651</td>
          <td>25.094308</td>
          <td>0.057722</td>
          <td>24.674908</td>
          <td>0.076209</td>
          <td>23.890421</td>
          <td>0.085831</td>
          <td>0.048945</td>
          <td>0.044909</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.492372</td>
          <td>0.316726</td>
          <td>26.705359</td>
          <td>0.145937</td>
          <td>25.998579</td>
          <td>0.127859</td>
          <td>26.012058</td>
          <td>0.240729</td>
          <td>25.077235</td>
          <td>0.237654</td>
          <td>0.042682</td>
          <td>0.027338</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.183747</td>
          <td>0.483252</td>
          <td>26.068819</td>
          <td>0.135868</td>
          <td>24.921929</td>
          <td>0.094734</td>
          <td>24.464072</td>
          <td>0.141543</td>
          <td>0.141275</td>
          <td>0.105956</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.969672</td>
          <td>0.536788</td>
          <td>28.782349</td>
          <td>0.810971</td>
          <td>27.503862</td>
          <td>0.284724</td>
          <td>26.389309</td>
          <td>0.178756</td>
          <td>25.685388</td>
          <td>0.183199</td>
          <td>25.276567</td>
          <td>0.279791</td>
          <td>0.125472</td>
          <td>0.121678</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.139342</td>
          <td>0.283213</td>
          <td>26.063978</td>
          <td>0.094777</td>
          <td>25.837956</td>
          <td>0.068321</td>
          <td>25.709154</td>
          <td>0.099350</td>
          <td>25.607632</td>
          <td>0.171506</td>
          <td>25.021178</td>
          <td>0.226873</td>
          <td>0.132289</td>
          <td>0.128353</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>27.061479</td>
          <td>0.573489</td>
          <td>26.261897</td>
          <td>0.112674</td>
          <td>25.516365</td>
          <td>0.051363</td>
          <td>25.124950</td>
          <td>0.059314</td>
          <td>24.754192</td>
          <td>0.081734</td>
          <td>24.803362</td>
          <td>0.189054</td>
          <td>0.076085</td>
          <td>0.059943</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.677834</td>
          <td>0.869372</td>
          <td>26.914780</td>
          <td>0.197175</td>
          <td>26.102361</td>
          <td>0.086294</td>
          <td>25.272834</td>
          <td>0.067624</td>
          <td>24.848336</td>
          <td>0.088801</td>
          <td>24.110063</td>
          <td>0.104082</td>
          <td>0.101686</td>
          <td>0.100505</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.102516</td>
          <td>0.590500</td>
          <td>26.850469</td>
          <td>0.186777</td>
          <td>26.590831</td>
          <td>0.132213</td>
          <td>26.512120</td>
          <td>0.198284</td>
          <td>26.110086</td>
          <td>0.260919</td>
          <td>25.406480</td>
          <td>0.310670</td>
          <td>0.088952</td>
          <td>0.083788</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.812616</td>
          <td>0.478262</td>
          <td>26.269902</td>
          <td>0.113462</td>
          <td>25.958966</td>
          <td>0.076040</td>
          <td>25.891199</td>
          <td>0.116475</td>
          <td>25.953856</td>
          <td>0.229414</td>
          <td>25.087359</td>
          <td>0.239649</td>
          <td>0.017071</td>
          <td>0.016200</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.760641</td>
          <td>0.460051</td>
          <td>26.746338</td>
          <td>0.171006</td>
          <td>26.466557</td>
          <td>0.118704</td>
          <td>26.288206</td>
          <td>0.164027</td>
          <td>25.851631</td>
          <td>0.210695</td>
          <td>25.223236</td>
          <td>0.267915</td>
          <td>0.164908</td>
          <td>0.084711</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model_gaap


.. parsed-literal::

    Inserting handle into data store.  output_error_model_gaap: inprogress_output_error_model_gaap.pq, error_model_gaap




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>29.414679</td>
          <td>2.245181</td>
          <td>26.396699</td>
          <td>0.146728</td>
          <td>26.065847</td>
          <td>0.098971</td>
          <td>25.073325</td>
          <td>0.067703</td>
          <td>24.498386</td>
          <td>0.077290</td>
          <td>24.075642</td>
          <td>0.120126</td>
          <td>0.048945</td>
          <td>0.044909</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.147994</td>
          <td>0.670833</td>
          <td>28.147947</td>
          <td>0.590163</td>
          <td>26.656084</td>
          <td>0.164461</td>
          <td>26.263783</td>
          <td>0.189854</td>
          <td>26.039208</td>
          <td>0.287114</td>
          <td>24.641952</td>
          <td>0.194491</td>
          <td>0.042682</td>
          <td>0.027338</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.833310</td>
          <td>1.065069</td>
          <td>28.944924</td>
          <td>1.025270</td>
          <td>27.485276</td>
          <td>0.340128</td>
          <td>26.114450</td>
          <td>0.175130</td>
          <td>24.982740</td>
          <td>0.123288</td>
          <td>24.279218</td>
          <td>0.149532</td>
          <td>0.141275</td>
          <td>0.105956</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.823729</td>
          <td>1.058635</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.376293</td>
          <td>0.311674</td>
          <td>26.178410</td>
          <td>0.184736</td>
          <td>25.410332</td>
          <td>0.177819</td>
          <td>25.510402</td>
          <td>0.409248</td>
          <td>0.125472</td>
          <td>0.121678</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.703049</td>
          <td>0.504347</td>
          <td>26.077795</td>
          <td>0.116185</td>
          <td>25.831879</td>
          <td>0.084482</td>
          <td>25.759258</td>
          <td>0.129725</td>
          <td>25.168108</td>
          <td>0.145330</td>
          <td>25.170100</td>
          <td>0.314998</td>
          <td>0.132289</td>
          <td>0.128353</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>26.392416</td>
          <td>0.388774</td>
          <td>26.423118</td>
          <td>0.151188</td>
          <td>25.515252</td>
          <td>0.061399</td>
          <td>25.114395</td>
          <td>0.070816</td>
          <td>24.942759</td>
          <td>0.115082</td>
          <td>25.507968</td>
          <td>0.396311</td>
          <td>0.076085</td>
          <td>0.059943</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.589337</td>
          <td>1.583448</td>
          <td>26.701638</td>
          <td>0.194568</td>
          <td>25.913026</td>
          <td>0.088841</td>
          <td>25.134917</td>
          <td>0.073472</td>
          <td>24.754761</td>
          <td>0.099412</td>
          <td>24.195128</td>
          <td>0.136797</td>
          <td>0.101686</td>
          <td>0.100505</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.274852</td>
          <td>1.345220</td>
          <td>26.737910</td>
          <td>0.199059</td>
          <td>26.322764</td>
          <td>0.125964</td>
          <td>26.739479</td>
          <td>0.287021</td>
          <td>25.526895</td>
          <td>0.191641</td>
          <td>25.998768</td>
          <td>0.575447</td>
          <td>0.088952</td>
          <td>0.083788</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.302261</td>
          <td>0.358849</td>
          <td>26.164579</td>
          <td>0.119385</td>
          <td>26.065435</td>
          <td>0.098293</td>
          <td>25.999585</td>
          <td>0.151114</td>
          <td>25.903793</td>
          <td>0.256336</td>
          <td>25.244502</td>
          <td>0.317950</td>
          <td>0.017071</td>
          <td>0.016200</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.382016</td>
          <td>0.395948</td>
          <td>26.758263</td>
          <td>0.207742</td>
          <td>27.056646</td>
          <td>0.241454</td>
          <td>26.176164</td>
          <td>0.185274</td>
          <td>26.360748</td>
          <td>0.387397</td>
          <td>25.055869</td>
          <td>0.287265</td>
          <td>0.164908</td>
          <td>0.084711</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



.. parsed-literal::

    Inserting handle into data store.  output_truth: None, error_model_auto


.. parsed-literal::

    Inserting handle into data store.  output_error_model_auto: inprogress_output_error_model_auto.pq, error_model_auto




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>redshift</th>
          <th>u</th>
          <th>u_err</th>
          <th>g</th>
          <th>g_err</th>
          <th>r</th>
          <th>r_err</th>
          <th>i</th>
          <th>i_err</th>
          <th>z</th>
          <th>z_err</th>
          <th>y</th>
          <th>y_err</th>
          <th>major</th>
          <th>minor</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>1.398944</td>
          <td>28.683337</td>
          <td>1.548690</td>
          <td>27.140058</td>
          <td>0.243612</td>
          <td>26.013784</td>
          <td>0.082180</td>
          <td>25.079471</td>
          <td>0.058754</td>
          <td>24.750826</td>
          <td>0.083905</td>
          <td>23.880112</td>
          <td>0.087671</td>
          <td>0.048945</td>
          <td>0.044909</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.284514</td>
          <td>0.676777</td>
          <td>27.641811</td>
          <td>0.361236</td>
          <td>26.414779</td>
          <td>0.115413</td>
          <td>25.914562</td>
          <td>0.120996</td>
          <td>25.951469</td>
          <td>0.232678</td>
          <td>25.481691</td>
          <td>0.335189</td>
          <td>0.042682</td>
          <td>0.027338</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>29.435022</td>
          <td>2.266841</td>
          <td>29.240489</td>
          <td>1.189007</td>
          <td>28.072222</td>
          <td>0.514903</td>
          <td>26.148929</td>
          <td>0.173595</td>
          <td>25.181172</td>
          <td>0.140979</td>
          <td>24.340266</td>
          <td>0.151665</td>
          <td>0.141275</td>
          <td>0.105956</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.062054</td>
          <td>0.557676</td>
          <td>26.878723</td>
          <td>0.199824</td>
          <td>26.379025</td>
          <td>0.210502</td>
          <td>25.224873</td>
          <td>0.146202</td>
          <td>25.052060</td>
          <td>0.274860</td>
          <td>0.125472</td>
          <td>0.121678</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>27.271148</td>
          <td>0.738894</td>
          <td>26.010137</td>
          <td>0.106823</td>
          <td>25.915343</td>
          <td>0.088419</td>
          <td>25.741094</td>
          <td>0.124131</td>
          <td>25.403082</td>
          <td>0.172917</td>
          <td>25.165863</td>
          <td>0.305883</td>
          <td>0.132289</td>
          <td>0.128353</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>99995</th>
          <td>0.389450</td>
          <td>28.959339</td>
          <td>1.784424</td>
          <td>26.188788</td>
          <td>0.111342</td>
          <td>25.421616</td>
          <td>0.050163</td>
          <td>24.980722</td>
          <td>0.055596</td>
          <td>24.803737</td>
          <td>0.090641</td>
          <td>24.768689</td>
          <td>0.194838</td>
          <td>0.076085</td>
          <td>0.059943</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>29.963875</td>
          <td>2.701476</td>
          <td>26.458637</td>
          <td>0.148479</td>
          <td>26.016852</td>
          <td>0.090450</td>
          <td>25.250419</td>
          <td>0.075384</td>
          <td>24.957522</td>
          <td>0.110354</td>
          <td>24.610245</td>
          <td>0.181309</td>
          <td>0.101686</td>
          <td>0.100505</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.210787</td>
          <td>0.670882</td>
          <td>26.779337</td>
          <td>0.190080</td>
          <td>26.281171</td>
          <td>0.110656</td>
          <td>26.248957</td>
          <td>0.174137</td>
          <td>25.708133</td>
          <td>0.203943</td>
          <td>26.013655</td>
          <td>0.537187</td>
          <td>0.088952</td>
          <td>0.083788</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.630018</td>
          <td>0.417688</td>
          <td>26.023931</td>
          <td>0.091804</td>
          <td>26.014070</td>
          <td>0.080134</td>
          <td>25.887742</td>
          <td>0.116579</td>
          <td>25.493673</td>
          <td>0.156182</td>
          <td>25.981885</td>
          <td>0.486233</td>
          <td>0.017071</td>
          <td>0.016200</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.918354</td>
          <td>0.267940</td>
          <td>26.720694</td>
          <td>0.195493</td>
          <td>26.711780</td>
          <td>0.175037</td>
          <td>26.237282</td>
          <td>0.188464</td>
          <td>26.205624</td>
          <td>0.332675</td>
          <td>26.535091</td>
          <td>0.825427</td>
          <td>0.164908</td>
          <td>0.084711</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



Notice some of the magnitudes are inf’s. These are non-detections
(i.e. the noisy flux was negative). You can change the nSigma limit for
non-detections by setting ``sigLim=...``. For example, if ``sigLim=5``,
then all fluxes with ``SNR<5`` are flagged as non-detections.

Let’s plot the error as a function of magnitude

.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_gaap.data[band].to_numpy(),
                samples_w_errs_gaap.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='GAAP')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_22_0.png


.. code:: ipython3

    %matplotlib inline
    
    fig, axes_ = plt.subplots(ncols=3, nrows=2, figsize=(15, 9), dpi=100)
    axes = axes_.reshape(-1)
    for i, band in enumerate("ugrizy"):
        ax = axes[i]
        # pull out the magnitudes and errors
        mags = samples_w_errs.data[band].to_numpy()
        errs = samples_w_errs.data[band + "_err"].to_numpy()
        
        # sort them by magnitude
        mags, errs = mags[mags.argsort()], errs[mags.argsort()]
        
        # plot errs vs mags
        #ax.plot(mags, errs, label=band) 
        
        #plt.plot(mags, errs, c='C'+str(i))
        ax.scatter(samples_w_errs_auto.data[band].to_numpy(),
                samples_w_errs_auto.data[band + "_err"].to_numpy(),
                    s=5, marker='.', color='C0', alpha=0.8, label='AUTO')
        
        ax.plot(mags, errs, color='C3', label='Point source')
        
        
        ax.legend()
        ax.set_xlim(18, 31)
        ax.set_ylim(-0.1, 3.5)
        ax.set(xlabel=band+" Band Magnitude (AB)", ylabel="Error (mags)")




.. image:: 01_Photometric_Realization_files/01_Photometric_Realization_23_0.png


You can see that the photometric error increases as magnitude gets
dimmer, just like you would expect, and that the extended source errors
are greater than the point source errors. The extended source errors are
also scattered, because the galaxies have random sizes.

Also, you can find the GAaP and AUTO magnitude error are scattered due
to variable galaxy sizes. Also, you can find that there are gaps between
GAAP magnitude error and point souce magnitude error, this is because
the additional factors due to aperture sizes have a minimum value of
:math:`\sqrt{(\sigma^2+A_{\mathrm{min}})/\sigma^2}`, where
:math:`\sigma` is the width of the beam, :math:`A_{\min}` is an offset
of the aperture sizes (taken to be 0.7 arcmin here).

You can also see that there are *very* faint galaxies in this sample.
That’s because, by default, the error model returns magnitudes for all
positive fluxes. If you want these galaxies flagged as non-detections
instead, you can set e.g. ``sigLim=5``, and everything with ``SNR<5``
will be flagged as a non-detection.
