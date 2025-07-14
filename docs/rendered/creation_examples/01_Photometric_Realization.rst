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

    Inserting handle into data store.  model: /opt/hostedtoolcache/Python/3.10.18/x64/lib/python3.10/site-packages/pzflow/example_files/example-flow.pzflow.pkl, truth


Let’s check that the Engine correctly read the underlying PZ Flow
object:

.. code:: ipython3

    flowEngine_truth.get_data("model")





.. parsed-literal::

    <pzflow.flow.Flow at 0x7f2f34d24640>



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
    0      23.994413  0.083475  0.055181  
    1      25.391064  0.117854  0.085509  
    2      24.304707  0.013093  0.007906  
    3      25.291103  0.043386  0.028841  
    4      25.096743  0.055540  0.040257  
    ...          ...       ...       ...  
    99995  24.737946  0.160805  0.152299  
    99996  24.224169  0.132924  0.113014  
    99997  25.613836  0.105207  0.058660  
    99998  25.274899  0.047719  0.039312  
    99999  25.699642  0.031673  0.022696  
    
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
          <td>inf</td>
          <td>inf</td>
          <td>26.550620</td>
          <td>0.144658</td>
          <td>26.002025</td>
          <td>0.078988</td>
          <td>25.054760</td>
          <td>0.055731</td>
          <td>24.653543</td>
          <td>0.074784</td>
          <td>24.209833</td>
          <td>0.113555</td>
          <td>0.083475</td>
          <td>0.055181</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.917533</td>
          <td>1.007860</td>
          <td>27.656743</td>
          <td>0.360678</td>
          <td>26.463734</td>
          <td>0.118413</td>
          <td>26.245636</td>
          <td>0.158169</td>
          <td>25.676968</td>
          <td>0.181899</td>
          <td>25.202069</td>
          <td>0.263326</td>
          <td>0.117854</td>
          <td>0.085509</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>27.372118</td>
          <td>0.711732</td>
          <td>28.946553</td>
          <td>0.900431</td>
          <td>27.903542</td>
          <td>0.390737</td>
          <td>26.006172</td>
          <td>0.128703</td>
          <td>25.075201</td>
          <td>0.108340</td>
          <td>24.438984</td>
          <td>0.138516</td>
          <td>0.013093</td>
          <td>0.007906</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.933602</td>
          <td>0.446271</td>
          <td>27.858998</td>
          <td>0.377473</td>
          <td>26.403399</td>
          <td>0.180903</td>
          <td>25.195758</td>
          <td>0.120338</td>
          <td>25.854106</td>
          <td>0.440360</td>
          <td>0.043386</td>
          <td>0.028841</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.059753</td>
          <td>0.265492</td>
          <td>26.140916</td>
          <td>0.101383</td>
          <td>25.907358</td>
          <td>0.072649</td>
          <td>25.689411</td>
          <td>0.097645</td>
          <td>25.561545</td>
          <td>0.164905</td>
          <td>25.052105</td>
          <td>0.232765</td>
          <td>0.055540</td>
          <td>0.040257</td>
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
          <td>27.023510</td>
          <td>0.558085</td>
          <td>26.215169</td>
          <td>0.108177</td>
          <td>25.371232</td>
          <td>0.045153</td>
          <td>25.130657</td>
          <td>0.059615</td>
          <td>24.854735</td>
          <td>0.089302</td>
          <td>25.022289</td>
          <td>0.227082</td>
          <td>0.160805</td>
          <td>0.152299</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.999720</td>
          <td>0.548596</td>
          <td>26.667566</td>
          <td>0.159905</td>
          <td>25.966526</td>
          <td>0.076550</td>
          <td>25.183604</td>
          <td>0.062481</td>
          <td>24.748756</td>
          <td>0.081343</td>
          <td>24.177017</td>
          <td>0.110352</td>
          <td>0.132924</td>
          <td>0.113014</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.338620</td>
          <td>0.695766</td>
          <td>26.505607</td>
          <td>0.139162</td>
          <td>26.666821</td>
          <td>0.141176</td>
          <td>26.371517</td>
          <td>0.176078</td>
          <td>25.777096</td>
          <td>0.197932</td>
          <td>26.739445</td>
          <td>0.821048</td>
          <td>0.105207</td>
          <td>0.058660</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.313041</td>
          <td>0.325524</td>
          <td>26.097351</td>
          <td>0.097590</td>
          <td>26.122064</td>
          <td>0.087804</td>
          <td>25.771569</td>
          <td>0.104930</td>
          <td>25.720670</td>
          <td>0.188744</td>
          <td>25.687183</td>
          <td>0.387533</td>
          <td>0.047719</td>
          <td>0.039312</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>25.935877</td>
          <td>0.239863</td>
          <td>26.778989</td>
          <td>0.175813</td>
          <td>26.750932</td>
          <td>0.151761</td>
          <td>26.782673</td>
          <td>0.248327</td>
          <td>25.883216</td>
          <td>0.216326</td>
          <td>25.495966</td>
          <td>0.333618</td>
          <td>0.031673</td>
          <td>0.022696</td>
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
          <td>27.558141</td>
          <td>0.885000</td>
          <td>26.797881</td>
          <td>0.207866</td>
          <td>25.902317</td>
          <td>0.086529</td>
          <td>25.072561</td>
          <td>0.068315</td>
          <td>24.708508</td>
          <td>0.093864</td>
          <td>23.992401</td>
          <td>0.112792</td>
          <td>0.083475</td>
          <td>0.055181</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.196561</td>
          <td>0.292903</td>
          <td>26.558213</td>
          <td>0.155802</td>
          <td>26.164993</td>
          <td>0.179999</td>
          <td>25.826420</td>
          <td>0.248266</td>
          <td>25.059863</td>
          <td>0.282925</td>
          <td>0.117854</td>
          <td>0.085509</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.957747</td>
          <td>0.585915</td>
          <td>29.043040</td>
          <td>1.054098</td>
          <td>27.317697</td>
          <td>0.284268</td>
          <td>25.820402</td>
          <td>0.129420</td>
          <td>25.085480</td>
          <td>0.128308</td>
          <td>24.534237</td>
          <td>0.176877</td>
          <td>0.013093</td>
          <td>0.007906</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>29.294335</td>
          <td>2.138694</td>
          <td>28.341741</td>
          <td>0.675745</td>
          <td>28.140343</td>
          <td>0.537769</td>
          <td>26.487540</td>
          <td>0.228993</td>
          <td>25.296011</td>
          <td>0.154452</td>
          <td>25.109264</td>
          <td>0.286199</td>
          <td>0.043386</td>
          <td>0.028841</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.085025</td>
          <td>0.303565</td>
          <td>26.109564</td>
          <td>0.114526</td>
          <td>26.034177</td>
          <td>0.096307</td>
          <td>25.634574</td>
          <td>0.110977</td>
          <td>25.418430</td>
          <td>0.172016</td>
          <td>25.044675</td>
          <td>0.272452</td>
          <td>0.055540</td>
          <td>0.040257</td>
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
          <td>28.425943</td>
          <td>1.488914</td>
          <td>26.499450</td>
          <td>0.170357</td>
          <td>25.440439</td>
          <td>0.061138</td>
          <td>25.008760</td>
          <td>0.068752</td>
          <td>24.847471</td>
          <td>0.112557</td>
          <td>24.911875</td>
          <td>0.261098</td>
          <td>0.160805</td>
          <td>0.152299</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.926027</td>
          <td>0.590492</td>
          <td>26.750568</td>
          <td>0.205484</td>
          <td>25.885888</td>
          <td>0.088111</td>
          <td>25.147276</td>
          <td>0.075491</td>
          <td>24.855367</td>
          <td>0.110246</td>
          <td>24.066760</td>
          <td>0.124360</td>
          <td>0.132924</td>
          <td>0.113014</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>inf</td>
          <td>inf</td>
          <td>26.818830</td>
          <td>0.212858</td>
          <td>26.202530</td>
          <td>0.113372</td>
          <td>26.181345</td>
          <td>0.180562</td>
          <td>26.153472</td>
          <td>0.320374</td>
          <td>25.359385</td>
          <td>0.355696</td>
          <td>0.105207</td>
          <td>0.058660</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.931840</td>
          <td>0.267956</td>
          <td>26.219052</td>
          <td>0.125777</td>
          <td>26.202473</td>
          <td>0.111415</td>
          <td>25.843481</td>
          <td>0.132850</td>
          <td>25.638215</td>
          <td>0.206775</td>
          <td>25.086291</td>
          <td>0.281423</td>
          <td>0.047719</td>
          <td>0.039312</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.135453</td>
          <td>0.664373</td>
          <td>26.660972</td>
          <td>0.183005</td>
          <td>26.541758</td>
          <td>0.148875</td>
          <td>26.038450</td>
          <td>0.156489</td>
          <td>25.574443</td>
          <td>0.195271</td>
          <td>25.271848</td>
          <td>0.325447</td>
          <td>0.031673</td>
          <td>0.022696</td>
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
          <td>29.403590</td>
          <td>2.158215</td>
          <td>26.787080</td>
          <td>0.186688</td>
          <td>26.095042</td>
          <td>0.091335</td>
          <td>25.198660</td>
          <td>0.067689</td>
          <td>24.533427</td>
          <td>0.071657</td>
          <td>24.089869</td>
          <td>0.109137</td>
          <td>0.083475</td>
          <td>0.055181</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>26.787119</td>
          <td>0.506204</td>
          <td>27.896341</td>
          <td>0.476644</td>
          <td>26.394558</td>
          <td>0.126143</td>
          <td>26.316616</td>
          <td>0.190530</td>
          <td>25.690724</td>
          <td>0.207348</td>
          <td>25.520129</td>
          <td>0.381473</td>
          <td>0.117854</td>
          <td>0.085509</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>28.399160</td>
          <td>1.324469</td>
          <td>inf</td>
          <td>inf</td>
          <td>28.338889</td>
          <td>0.542254</td>
          <td>25.969546</td>
          <td>0.124887</td>
          <td>24.877983</td>
          <td>0.091291</td>
          <td>24.324385</td>
          <td>0.125650</td>
          <td>0.013093</td>
          <td>0.007906</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.650790</td>
          <td>1.517428</td>
          <td>28.127047</td>
          <td>0.522118</td>
          <td>27.447447</td>
          <td>0.276559</td>
          <td>26.510224</td>
          <td>0.201586</td>
          <td>25.235083</td>
          <td>0.126754</td>
          <td>25.627698</td>
          <td>0.376226</td>
          <td>0.043386</td>
          <td>0.028841</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.355925</td>
          <td>0.343370</td>
          <td>26.225056</td>
          <td>0.112066</td>
          <td>25.911133</td>
          <td>0.075195</td>
          <td>25.682887</td>
          <td>0.100277</td>
          <td>25.161576</td>
          <td>0.120435</td>
          <td>25.215339</td>
          <td>0.274218</td>
          <td>0.055540</td>
          <td>0.040257</td>
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
          <td>27.460231</td>
          <td>0.867370</td>
          <td>26.290051</td>
          <td>0.144263</td>
          <td>25.468172</td>
          <td>0.063567</td>
          <td>24.997191</td>
          <td>0.069066</td>
          <td>25.016208</td>
          <td>0.132155</td>
          <td>24.762794</td>
          <td>0.234137</td>
          <td>0.160805</td>
          <td>0.152299</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>26.752221</td>
          <td>0.508049</td>
          <td>26.693952</td>
          <td>0.189369</td>
          <td>25.917874</td>
          <td>0.087117</td>
          <td>25.198829</td>
          <td>0.075835</td>
          <td>25.026046</td>
          <td>0.123004</td>
          <td>24.158203</td>
          <td>0.129350</td>
          <td>0.132924</td>
          <td>0.113014</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>27.036658</td>
          <td>0.592819</td>
          <td>26.939920</td>
          <td>0.216719</td>
          <td>26.298541</td>
          <td>0.111867</td>
          <td>26.312726</td>
          <td>0.183003</td>
          <td>25.774704</td>
          <td>0.214743</td>
          <td>24.969270</td>
          <td>0.236745</td>
          <td>0.105207</td>
          <td>0.058660</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.697901</td>
          <td>0.445521</td>
          <td>26.118848</td>
          <td>0.101649</td>
          <td>26.150776</td>
          <td>0.092359</td>
          <td>25.794274</td>
          <td>0.109894</td>
          <td>26.007029</td>
          <td>0.245478</td>
          <td>24.845808</td>
          <td>0.200920</td>
          <td>0.047719</td>
          <td>0.039312</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>26.589691</td>
          <td>0.406580</td>
          <td>26.622139</td>
          <td>0.155146</td>
          <td>26.422345</td>
          <td>0.115384</td>
          <td>26.275203</td>
          <td>0.163913</td>
          <td>26.436054</td>
          <td>0.342286</td>
          <td>25.696662</td>
          <td>0.394038</td>
          <td>0.031673</td>
          <td>0.022696</td>
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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_24_0.png


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




.. image:: ../../../docs/rendered/creation_examples/01_Photometric_Realization_files/../../../docs/rendered/creation_examples/01_Photometric_Realization_25_0.png


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
