Photometric Realization from Different Magnitude Error Models
=============================================================

author: John Franklin Crenshaw, Sam Schmidt, Eric Charles, Ziang Yan

last run successfully: August 2, 2023

This notebook demonstrates how to do photometric realization from
different magnitude error models. For more completed degrader demo, see
``degradation-demo.ipynb``

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


We’ll start by setting up the RAIL data store. RAIL uses
`ceci <https://github.com/LSSTDESC/ceci>`__, which is designed for
pipelines rather than interactive notebooks, the data store will work
around that and enable us to use data interactively. See the
``rail/examples/goldenspike_examples/goldenspike.ipynb`` example
notebook for more details on the Data Store.

.. code:: ipython3

    DS = RailStage.data_store
    DS.__class__.allow_overwrite = True


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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.16/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f9439af6a10>



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
    0      23.994413  0.003319  0.002869  
    1      25.391064  0.008733  0.007945  
    2      24.304707  0.103938  0.052162  
    3      25.291103  0.147522  0.143359  
    4      25.096743  0.010929  0.009473  
    ...          ...       ...       ...  
    99995  24.737946  0.086491  0.071701  
    99996  24.224169  0.044537  0.022302  
    99997  25.613836  0.073146  0.047825  
    99998  25.274899  0.100551  0.094662  
    99999  25.699642  0.059611  0.049181  
    
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
          <td>31.643268</td>
          <td>4.209827</td>
          <td>26.572654</td>
          <td>0.147423</td>
          <td>26.128335</td>
          <td>0.088290</td>
          <td>25.111098</td>
          <td>0.058589</td>
          <td>24.628309</td>
          <td>0.073134</td>
          <td>24.071102</td>
          <td>0.100593</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.105431</td>
          <td>0.231178</td>
          <td>26.985483</td>
          <td>0.185318</td>
          <td>25.957344</td>
          <td>0.123368</td>
          <td>25.762016</td>
          <td>0.195437</td>
          <td>25.381250</td>
          <td>0.304451</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>30.318097</td>
          <td>1.864253</td>
          <td>28.773727</td>
          <td>0.733495</td>
          <td>26.016458</td>
          <td>0.129855</td>
          <td>25.049591</td>
          <td>0.105943</td>
          <td>24.255409</td>
          <td>0.118151</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.423288</td>
          <td>1.340639</td>
          <td>27.914558</td>
          <td>0.439894</td>
          <td>27.619990</td>
          <td>0.312611</td>
          <td>26.577584</td>
          <td>0.209474</td>
          <td>25.776909</td>
          <td>0.197901</td>
          <td>24.954974</td>
          <td>0.214709</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.896987</td>
          <td>0.232286</td>
          <td>26.241202</td>
          <td>0.110660</td>
          <td>26.064507</td>
          <td>0.083464</td>
          <td>25.615836</td>
          <td>0.091537</td>
          <td>25.488074</td>
          <td>0.154869</td>
          <td>24.799791</td>
          <td>0.188485</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>26.997710</td>
          <td>0.547799</td>
          <td>26.353208</td>
          <td>0.121981</td>
          <td>25.520540</td>
          <td>0.051554</td>
          <td>25.037871</td>
          <td>0.054902</td>
          <td>24.852808</td>
          <td>0.089151</td>
          <td>24.680496</td>
          <td>0.170360</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.143299</td>
          <td>0.607780</td>
          <td>26.776249</td>
          <td>0.175405</td>
          <td>26.116380</td>
          <td>0.087366</td>
          <td>25.100546</td>
          <td>0.058043</td>
          <td>24.873485</td>
          <td>0.090787</td>
          <td>24.202577</td>
          <td>0.112839</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.582399</td>
          <td>0.148661</td>
          <td>26.233104</td>
          <td>0.096803</td>
          <td>25.995448</td>
          <td>0.127513</td>
          <td>26.675516</td>
          <td>0.408725</td>
          <td>25.690181</td>
          <td>0.388434</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>27.197903</td>
          <td>0.631502</td>
          <td>26.153341</td>
          <td>0.102490</td>
          <td>26.192661</td>
          <td>0.093427</td>
          <td>25.981525</td>
          <td>0.125983</td>
          <td>25.547080</td>
          <td>0.162883</td>
          <td>26.100023</td>
          <td>0.528648</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.984824</td>
          <td>0.249713</td>
          <td>26.581843</td>
          <td>0.148590</td>
          <td>26.553189</td>
          <td>0.127974</td>
          <td>26.677154</td>
          <td>0.227595</td>
          <td>25.775619</td>
          <td>0.197686</td>
          <td>25.789987</td>
          <td>0.419410</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_gaap = errorModel_gaap(samples_truth)
    samples_w_errs_gaap.data



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
          <td>inf</td>
          <td>inf</td>
          <td>26.798169</td>
          <td>0.204950</td>
          <td>26.009036</td>
          <td>0.093463</td>
          <td>25.081731</td>
          <td>0.067681</td>
          <td>24.608550</td>
          <td>0.084541</td>
          <td>24.122984</td>
          <td>0.124224</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>29.152659</td>
          <td>2.014759</td>
          <td>27.650207</td>
          <td>0.407124</td>
          <td>26.786807</td>
          <td>0.183045</td>
          <td>26.349493</td>
          <td>0.203215</td>
          <td>25.760870</td>
          <td>0.227685</td>
          <td>25.834337</td>
          <td>0.500139</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.592296</td>
          <td>0.454762</td>
          <td>28.216715</td>
          <td>0.627590</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.063227</td>
          <td>0.163046</td>
          <td>24.831097</td>
          <td>0.105089</td>
          <td>24.073740</td>
          <td>0.121753</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>26.875969</td>
          <td>0.576330</td>
          <td>32.519720</td>
          <td>4.107547</td>
          <td>27.588092</td>
          <td>0.374314</td>
          <td>26.557042</td>
          <td>0.257645</td>
          <td>25.212030</td>
          <td>0.152769</td>
          <td>25.598208</td>
          <td>0.444478</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.561935</td>
          <td>0.438104</td>
          <td>26.220810</td>
          <td>0.125285</td>
          <td>25.910100</td>
          <td>0.085704</td>
          <td>25.733360</td>
          <td>0.120005</td>
          <td>25.353084</td>
          <td>0.161509</td>
          <td>24.895820</td>
          <td>0.239410</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>inf</td>
          <td>inf</td>
          <td>26.263991</td>
          <td>0.132468</td>
          <td>25.420192</td>
          <td>0.056741</td>
          <td>25.138086</td>
          <td>0.072717</td>
          <td>24.818691</td>
          <td>0.103821</td>
          <td>24.475114</td>
          <td>0.171697</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.502436</td>
          <td>1.498173</td>
          <td>27.020274</td>
          <td>0.247320</td>
          <td>25.966312</td>
          <td>0.090397</td>
          <td>25.248665</td>
          <td>0.078784</td>
          <td>24.900664</td>
          <td>0.109677</td>
          <td>24.239663</td>
          <td>0.137996</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.756537</td>
          <td>0.200113</td>
          <td>26.545140</td>
          <td>0.150809</td>
          <td>26.102997</td>
          <td>0.167067</td>
          <td>26.567931</td>
          <td>0.437916</td>
          <td>25.109759</td>
          <td>0.288545</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.128125</td>
          <td>0.319390</td>
          <td>26.095397</td>
          <td>0.115535</td>
          <td>26.131834</td>
          <td>0.107388</td>
          <td>25.838870</td>
          <td>0.135726</td>
          <td>26.144405</td>
          <td>0.320234</td>
          <td>25.025262</td>
          <td>0.274240</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.145159</td>
          <td>0.318973</td>
          <td>26.906642</td>
          <td>0.226285</td>
          <td>26.511943</td>
          <td>0.146178</td>
          <td>26.409909</td>
          <td>0.215828</td>
          <td>25.606955</td>
          <td>0.202120</td>
          <td>25.515004</td>
          <td>0.396445</td>
          <td>0.059611</td>
          <td>0.049181</td>
        </tr>
      </tbody>
    </table>
    <p>100000 rows × 15 columns</p>
    </div>



.. code:: ipython3

    samples_w_errs_auto = errorModel_auto(samples_truth)
    samples_w_errs_auto.data



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
          <td>27.040045</td>
          <td>0.564795</td>
          <td>26.851066</td>
          <td>0.186892</td>
          <td>26.186245</td>
          <td>0.092914</td>
          <td>25.177066</td>
          <td>0.062129</td>
          <td>24.633388</td>
          <td>0.073473</td>
          <td>24.020999</td>
          <td>0.096283</td>
          <td>0.003319</td>
          <td>0.002869</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.829657</td>
          <td>0.484625</td>
          <td>27.269936</td>
          <td>0.264866</td>
          <td>26.446533</td>
          <td>0.116763</td>
          <td>26.137546</td>
          <td>0.144301</td>
          <td>25.574409</td>
          <td>0.166877</td>
          <td>25.507597</td>
          <td>0.337004</td>
          <td>0.008733</td>
          <td>0.007945</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.954544</td>
          <td>1.795625</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.117272</td>
          <td>0.154024</td>
          <td>25.124599</td>
          <td>0.122653</td>
          <td>24.111219</td>
          <td>0.113347</td>
          <td>0.103938</td>
          <td>0.052162</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.714854</td>
          <td>0.999698</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.810756</td>
          <td>0.887223</td>
          <td>26.295963</td>
          <td>0.206817</td>
          <td>25.758608</td>
          <td>0.241217</td>
          <td>24.829210</td>
          <td>0.240843</td>
          <td>0.147522</td>
          <td>0.143359</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.476284</td>
          <td>0.370458</td>
          <td>26.260640</td>
          <td>0.112688</td>
          <td>26.016816</td>
          <td>0.080140</td>
          <td>25.621414</td>
          <td>0.092124</td>
          <td>25.443007</td>
          <td>0.149204</td>
          <td>25.167621</td>
          <td>0.256359</td>
          <td>0.010929</td>
          <td>0.009473</td>
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
          <td>26.427881</td>
          <td>0.374534</td>
          <td>26.707784</td>
          <td>0.176908</td>
          <td>25.473334</td>
          <td>0.053542</td>
          <td>25.083689</td>
          <td>0.062150</td>
          <td>24.922681</td>
          <td>0.102539</td>
          <td>24.975088</td>
          <td>0.235827</td>
          <td>0.086491</td>
          <td>0.071701</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.850361</td>
          <td>0.975467</td>
          <td>26.832137</td>
          <td>0.186458</td>
          <td>26.047145</td>
          <td>0.083567</td>
          <td>25.142343</td>
          <td>0.061299</td>
          <td>24.897917</td>
          <td>0.094297</td>
          <td>24.140841</td>
          <td>0.108756</td>
          <td>0.044537</td>
          <td>0.022302</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>28.466788</td>
          <td>1.399878</td>
          <td>26.770293</td>
          <td>0.181864</td>
          <td>26.501649</td>
          <td>0.128435</td>
          <td>26.203745</td>
          <td>0.160406</td>
          <td>25.543119</td>
          <td>0.170212</td>
          <td>25.520516</td>
          <td>0.355978</td>
          <td>0.073146</td>
          <td>0.047825</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.044930</td>
          <td>0.283029</td>
          <td>26.062917</td>
          <td>0.104718</td>
          <td>26.124862</td>
          <td>0.098721</td>
          <td>25.773068</td>
          <td>0.118340</td>
          <td>25.507166</td>
          <td>0.175918</td>
          <td>26.170013</td>
          <td>0.612257</td>
          <td>0.100551</td>
          <td>0.094662</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.137964</td>
          <td>0.618853</td>
          <td>26.605897</td>
          <td>0.156786</td>
          <td>26.809470</td>
          <td>0.165710</td>
          <td>26.439992</td>
          <td>0.194041</td>
          <td>26.041334</td>
          <td>0.255730</td>
          <td>inf</td>
          <td>inf</td>
          <td>0.059611</td>
          <td>0.049181</td>
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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/photometric_realization_demo_files/../../../docs/rendered/creation_examples/photometric_realization_demo_25_0.png


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
