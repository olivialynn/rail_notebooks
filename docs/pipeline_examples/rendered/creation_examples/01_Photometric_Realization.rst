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

    <pzflow.flow.Flow at 0x7f1b7ccf35e0>



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
    0      23.994413  0.065769  0.046186  
    1      25.391064  0.046449  0.025264  
    2      24.304707  0.115140  0.064693  
    3      25.291103  0.020988  0.016749  
    4      25.096743  0.032071  0.017911  
    ...          ...       ...       ...  
    99995  24.737946  0.135249  0.133751  
    99996  24.224169  0.017805  0.013426  
    99997  25.613836  0.047638  0.038215  
    99998  25.274899  0.029008  0.017086  
    99999  25.699642  0.021545  0.014246  
    
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
          <td>26.922597</td>
          <td>0.518687</td>
          <td>26.976271</td>
          <td>0.207609</td>
          <td>26.025322</td>
          <td>0.080629</td>
          <td>25.195691</td>
          <td>0.063155</td>
          <td>24.658833</td>
          <td>0.075134</td>
          <td>23.865869</td>
          <td>0.083994</td>
          <td>0.065769</td>
          <td>0.046186</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>27.079316</td>
          <td>0.580837</td>
          <td>27.978439</td>
          <td>0.461580</td>
          <td>26.835146</td>
          <td>0.163100</td>
          <td>26.125719</td>
          <td>0.142701</td>
          <td>25.647796</td>
          <td>0.177457</td>
          <td>26.941325</td>
          <td>0.932787</td>
          <td>0.046449</td>
          <td>0.025264</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.656631</td>
          <td>0.321888</td>
          <td>26.077225</td>
          <td>0.136858</td>
          <td>24.995036</td>
          <td>0.101005</td>
          <td>24.593445</td>
          <td>0.158168</td>
          <td>0.115140</td>
          <td>0.064693</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>27.627684</td>
          <td>0.842042</td>
          <td>27.517090</td>
          <td>0.323027</td>
          <td>27.143199</td>
          <td>0.211591</td>
          <td>26.103197</td>
          <td>0.139958</td>
          <td>25.777694</td>
          <td>0.198031</td>
          <td>25.461678</td>
          <td>0.324659</td>
          <td>0.020988</td>
          <td>0.016749</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.998768</td>
          <td>0.548218</td>
          <td>26.232492</td>
          <td>0.109823</td>
          <td>26.004692</td>
          <td>0.079174</td>
          <td>25.735699</td>
          <td>0.101687</td>
          <td>25.708394</td>
          <td>0.186797</td>
          <td>25.102036</td>
          <td>0.242569</td>
          <td>0.032071</td>
          <td>0.017911</td>
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
          <td>26.972868</td>
          <td>0.538035</td>
          <td>26.302139</td>
          <td>0.116689</td>
          <td>25.436737</td>
          <td>0.047857</td>
          <td>25.104982</td>
          <td>0.058272</td>
          <td>24.869938</td>
          <td>0.090504</td>
          <td>24.880648</td>
          <td>0.201760</td>
          <td>0.135249</td>
          <td>0.133751</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>27.546928</td>
          <td>0.799242</td>
          <td>27.068184</td>
          <td>0.224145</td>
          <td>26.168831</td>
          <td>0.091491</td>
          <td>25.257582</td>
          <td>0.066716</td>
          <td>24.727842</td>
          <td>0.079856</td>
          <td>24.100432</td>
          <td>0.103209</td>
          <td>0.017805</td>
          <td>0.013426</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.956509</td>
          <td>0.531678</td>
          <td>26.630742</td>
          <td>0.154949</td>
          <td>26.370952</td>
          <td>0.109216</td>
          <td>26.293100</td>
          <td>0.164713</td>
          <td>26.044474</td>
          <td>0.247246</td>
          <td>25.736341</td>
          <td>0.402516</td>
          <td>0.047638</td>
          <td>0.038215</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.143079</td>
          <td>0.284070</td>
          <td>26.105519</td>
          <td>0.098290</td>
          <td>26.161000</td>
          <td>0.090864</td>
          <td>25.919928</td>
          <td>0.119422</td>
          <td>25.753688</td>
          <td>0.194072</td>
          <td>25.475359</td>
          <td>0.328209</td>
          <td>0.029008</td>
          <td>0.017086</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.098012</td>
          <td>0.588614</td>
          <td>26.979902</td>
          <td>0.208241</td>
          <td>26.549169</td>
          <td>0.127529</td>
          <td>26.531750</td>
          <td>0.201581</td>
          <td>25.674360</td>
          <td>0.181497</td>
          <td>26.478752</td>
          <td>0.690576</td>
          <td>0.021545</td>
          <td>0.014246</td>
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
          <td>29.344149</td>
          <td>2.186395</td>
          <td>26.682941</td>
          <td>0.187779</td>
          <td>26.149984</td>
          <td>0.106893</td>
          <td>25.129423</td>
          <td>0.071402</td>
          <td>24.645195</td>
          <td>0.088266</td>
          <td>23.933907</td>
          <td>0.106542</td>
          <td>0.065769</td>
          <td>0.046186</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.155750</td>
          <td>0.276396</td>
          <td>26.786626</td>
          <td>0.183812</td>
          <td>26.418277</td>
          <td>0.216204</td>
          <td>25.474408</td>
          <td>0.179826</td>
          <td>25.012084</td>
          <td>0.264499</td>
          <td>0.046449</td>
          <td>0.025264</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>inf</td>
          <td>inf</td>
          <td>31.065756</td>
          <td>2.684485</td>
          <td>28.078200</td>
          <td>0.524303</td>
          <td>25.944327</td>
          <td>0.148202</td>
          <td>24.848058</td>
          <td>0.107327</td>
          <td>24.300617</td>
          <td>0.149043</td>
          <td>0.115140</td>
          <td>0.064693</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.305321</td>
          <td>0.310905</td>
          <td>27.389964</td>
          <td>0.301559</td>
          <td>26.265262</td>
          <td>0.189501</td>
          <td>25.484746</td>
          <td>0.180796</td>
          <td>25.145003</td>
          <td>0.293642</td>
          <td>0.020988</td>
          <td>0.016749</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>26.250160</td>
          <td>0.344783</td>
          <td>26.285647</td>
          <td>0.132739</td>
          <td>25.934438</td>
          <td>0.087730</td>
          <td>25.678784</td>
          <td>0.114668</td>
          <td>25.399809</td>
          <td>0.168389</td>
          <td>25.179895</td>
          <td>0.302307</td>
          <td>0.032071</td>
          <td>0.017911</td>
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
          <td>26.751179</td>
          <td>0.523589</td>
          <td>26.382670</td>
          <td>0.151616</td>
          <td>25.429041</td>
          <td>0.059369</td>
          <td>25.003928</td>
          <td>0.067116</td>
          <td>24.843412</td>
          <td>0.110060</td>
          <td>24.423954</td>
          <td>0.170536</td>
          <td>0.135249</td>
          <td>0.133751</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.570198</td>
          <td>1.547063</td>
          <td>26.917118</td>
          <td>0.226473</td>
          <td>25.996883</td>
          <td>0.092546</td>
          <td>25.276968</td>
          <td>0.080495</td>
          <td>24.706598</td>
          <td>0.092230</td>
          <td>24.222409</td>
          <td>0.135497</td>
          <td>0.017805</td>
          <td>0.013426</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.359907</td>
          <td>0.376711</td>
          <td>26.779492</td>
          <td>0.202861</td>
          <td>26.279739</td>
          <td>0.119149</td>
          <td>26.300451</td>
          <td>0.196191</td>
          <td>26.552477</td>
          <td>0.430391</td>
          <td>26.524766</td>
          <td>0.811789</td>
          <td>0.047638</td>
          <td>0.038215</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>26.766037</td>
          <td>0.510612</td>
          <td>26.025294</td>
          <td>0.105846</td>
          <td>26.033605</td>
          <td>0.095682</td>
          <td>25.833865</td>
          <td>0.131141</td>
          <td>25.704278</td>
          <td>0.217566</td>
          <td>25.441608</td>
          <td>0.371764</td>
          <td>0.029008</td>
          <td>0.017086</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.468025</td>
          <td>0.828487</td>
          <td>26.712034</td>
          <td>0.190826</td>
          <td>26.664186</td>
          <td>0.165084</td>
          <td>26.315736</td>
          <td>0.197712</td>
          <td>25.644761</td>
          <td>0.206858</td>
          <td>26.428196</td>
          <td>0.758978</td>
          <td>0.021545</td>
          <td>0.014246</td>
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
          <td>27.059195</td>
          <td>0.586348</td>
          <td>26.579992</td>
          <td>0.153731</td>
          <td>26.063515</td>
          <td>0.086965</td>
          <td>25.011670</td>
          <td>0.056073</td>
          <td>24.601832</td>
          <td>0.074513</td>
          <td>23.811904</td>
          <td>0.083660</td>
          <td>0.065769</td>
          <td>0.046186</td>
        </tr>
        <tr>
          <th>1</th>
          <td>2.285624</td>
          <td>inf</td>
          <td>inf</td>
          <td>27.235065</td>
          <td>0.261114</td>
          <td>26.896049</td>
          <td>0.174892</td>
          <td>26.355553</td>
          <td>0.177001</td>
          <td>25.485406</td>
          <td>0.157328</td>
          <td>24.920168</td>
          <td>0.212408</td>
          <td>0.046449</td>
          <td>0.025264</td>
        </tr>
        <tr>
          <th>2</th>
          <td>1.495132</td>
          <td>26.309610</td>
          <td>0.347099</td>
          <td>29.093771</td>
          <td>1.048750</td>
          <td>28.015044</td>
          <td>0.465148</td>
          <td>26.225924</td>
          <td>0.172701</td>
          <td>25.045916</td>
          <td>0.117007</td>
          <td>24.221418</td>
          <td>0.127505</td>
          <td>0.115140</td>
          <td>0.064693</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.842594</td>
          <td>28.160492</td>
          <td>1.163639</td>
          <td>28.192620</td>
          <td>0.542466</td>
          <td>27.286890</td>
          <td>0.239519</td>
          <td>26.251371</td>
          <td>0.159740</td>
          <td>25.587648</td>
          <td>0.169412</td>
          <td>25.331286</td>
          <td>0.293817</td>
          <td>0.020988</td>
          <td>0.016749</td>
        </tr>
        <tr>
          <th>4</th>
          <td>1.588960</td>
          <td>25.809132</td>
          <td>0.217252</td>
          <td>26.243280</td>
          <td>0.111730</td>
          <td>25.801605</td>
          <td>0.066764</td>
          <td>25.683179</td>
          <td>0.098040</td>
          <td>25.052764</td>
          <td>0.107199</td>
          <td>24.922909</td>
          <td>0.210914</td>
          <td>0.032071</td>
          <td>0.017911</td>
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
          <td>26.956617</td>
          <td>0.598793</td>
          <td>26.120307</td>
          <td>0.118640</td>
          <td>25.423208</td>
          <td>0.057812</td>
          <td>25.037955</td>
          <td>0.067661</td>
          <td>24.827573</td>
          <td>0.106293</td>
          <td>25.058122</td>
          <td>0.283101</td>
          <td>0.135249</td>
          <td>0.133751</td>
        </tr>
        <tr>
          <th>99996</th>
          <td>1.481047</td>
          <td>28.248353</td>
          <td>1.221387</td>
          <td>26.976650</td>
          <td>0.208254</td>
          <td>26.048680</td>
          <td>0.082586</td>
          <td>25.243713</td>
          <td>0.066137</td>
          <td>24.835372</td>
          <td>0.088090</td>
          <td>24.446724</td>
          <td>0.139923</td>
          <td>0.017805</td>
          <td>0.013426</td>
        </tr>
        <tr>
          <th>99997</th>
          <td>2.023548</td>
          <td>26.525318</td>
          <td>0.390375</td>
          <td>26.512124</td>
          <td>0.142922</td>
          <td>26.218381</td>
          <td>0.097946</td>
          <td>26.256329</td>
          <td>0.163689</td>
          <td>25.839631</td>
          <td>0.213538</td>
          <td>25.802512</td>
          <td>0.432929</td>
          <td>0.047638</td>
          <td>0.038215</td>
        </tr>
        <tr>
          <th>99998</th>
          <td>1.548204</td>
          <td>25.763898</td>
          <td>0.209016</td>
          <td>26.416113</td>
          <td>0.129658</td>
          <td>26.193877</td>
          <td>0.094243</td>
          <td>25.706403</td>
          <td>0.099906</td>
          <td>25.548669</td>
          <td>0.164315</td>
          <td>25.202530</td>
          <td>0.265372</td>
          <td>0.029008</td>
          <td>0.017086</td>
        </tr>
        <tr>
          <th>99999</th>
          <td>1.739491</td>
          <td>27.833224</td>
          <td>0.959764</td>
          <td>26.427760</td>
          <td>0.130621</td>
          <td>26.373622</td>
          <td>0.109962</td>
          <td>26.472012</td>
          <td>0.192574</td>
          <td>25.979216</td>
          <td>0.235280</td>
          <td>25.322319</td>
          <td>0.291598</td>
          <td>0.021545</td>
          <td>0.014246</td>
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
