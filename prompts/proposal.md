\section{Threat Model}
\label{sec:threat_model}

In this work, we introduce a novel backdoor attack targeting neural network models trained on tabular data, with a particular emphasis on manipulating categorical features—a dimension that has been largely overlooked in prior research. Previous attacks predominantly focused on numerical columns, modifying them to implant backdoors. Our approach extends this paradigm by incorporating categorical features into the attack vector, thereby enhancing the potential effectiveness and stealth of the backdoor.

We consider a classification task where a neural network model \( F: \mathcal{X} \rightarrow \mathcal{Y} \) maps inputs from the feature space \( \mathcal{X} \subseteq \mathbb{R}^d \) to the label space \( \mathcal{Y} = \{1, 2, \dots, C\} \), where \( C \) is the number of classes. To validate the feasibility of our proposed attack, we focus on a simplified setup using the TabNet model on the Forest Cover Type dataset. Future work will extend this approach to a broader range of models and datasets.

\subsection{Attacker's Capabilities and Objectives}

We assume an attacker with full access to the training dataset \( D_{original} = \{(x_i, y_i)\}_{i=1}^N \) and complete control over the training process. This scenario is realistic in contexts where model training is outsourced to a potentially malicious cloud service provider. The attacker's objective is to implant a backdoor into the model such that any input modified with a specific trigger pattern will cause the model to predict a target label \( t \in \mathcal{Y} \), irrespective of the input's true label.

\section{Converting Categorical Features}
\label{sec:convert_cf}

To incorporate categorical features into the attack without introducing ambiguity, we implement a hierarchical mapping strategy with an adaptive \(\Delta r\) that ensures a unique numerical representation for each category, even when multiple categories share the same frequency. We call this mapping function \(Conv(.)\) which \(D = Conv(D_{original})\). Here we explain the steps for mapping process:

\subparagraph{a. Primary Frequency-Based Mapping: }

For each categorical feature \( j \) with unique values \( \mathcal{V}_j = \{ v_{j1}, v_{j2}, \dots, v_{jk_j} \} \), where \( k_j \) is the number of categories, perform the following:

\begin{enumerate}
    \item \textbf{Compute Frequencies:} Calculate the frequency \( c_{jl} \) of each category \( v_{jl} \) in the dataset \( D_{original} \).
    
    \item \textbf{Initial Mapping:} Assign \( r_{jl} \) using the original formula:
    \[
    r_{jl} = \frac{c_{\text{max}, j} - c_{jl}}{c_{\text{max}, j} - 1}, \quad \text{for} \quad l = 1, \dots, k_j,
    \]
    where \( c_{\text{max}, j} = \max_{1 \leq l \leq k_j} c_{jl} \).
\end{enumerate}

\subparagraph{b. Adaptive \(\Delta r\) Selection: }

To determine \(\Delta r\) precisely and avoid ambiguity, we follow a systematic approach based on the smallest decimal precision in the primary mapping.

\begin{enumerate}
    \item \textbf{Sort Unique \( r_{jl} \) Values:} Sort the unique \( r_{jl} \) values in ascending order.
    
    \item \textbf{Compute Minimum Difference:}
    \[
    \Delta r_{\text{min}} = \min_{i} \left( r_{jl}^{(i+1)} - r_{jl}^{(i)} \right)
    \]
    
    \item \textbf{Determine \( p \):} Identify the largest single decimal component in \(\Delta r_{\text{min}}\). Specifically, express \(\Delta r_{\text{min}}\) in decimal form and determine the smallest decimal place \( p \) where a non-zero digit occurs.
    
    \begin{itemize}
        \item \textbf{Definition of \( p \):} Let \(\Delta r_{\text{min}} = 0.d_1d_2\dots d_n\), where \( d_1 \) is the first non-zero digit. Then, \( p \) is the position of the first non-zero digit.
        
        \item \textbf{Example Criteria:}
        \begin{itemize}
            \item If \( \Delta r_{\text{min}} = 0.4 \) (first decimal place), set \( p = 1 \).
            \item If \( \Delta r_{\text{min}} = 0.04 \) (second decimal place), set \( p = 2 \).
            \item If \( \Delta r_{\text{min}} = 0.004 \) (third decimal place), set \( p = 3 \).
            \item Continue similarly for smaller differences.
        \end{itemize}
    \end{itemize}
    
    \item \textbf{Set \(\Delta r\):} Define \(\Delta r\) as:
    \[
    \Delta r = 10^{-(p + 1)}
    \]
    This ensures that \(\Delta r\) is one order of magnitude smaller than the smallest decimal precision in \(\Delta r_{\text{min}}\), thereby maintaining uniqueness without overlapping existing \( r_{jl} \) values.
    
    \begin{itemize}
        \item \textbf{Example:}
        \begin{itemize}
            \item If \( \Delta r_{\text{min}} = 0.4 \) (first decimal place), then \( p = 1 \) and \( \Delta r = 0.01 \).
            \item If \( \Delta r_{\text{min}} = 0.04 \) (second decimal place), then \( p = 2 \) and \( \Delta r = 0.001 \).
        \end{itemize}
    \end{itemize}
\end{enumerate}

\subparagraph{c. Identifying and Resolving Ties: }

\begin{enumerate}
    \item \textbf{Detect Tied Categories:} For each feature \( j \), identify sets of categories that share the same frequency \( c_{jl} \).
    
    \item \textbf{Apply Secondary Ordering:} For each set of tied categories, apply a deterministic secondary ordering criterion, such as alphabetical order.
    
    \item \textbf{Assign Unique Offsets:} For each category \( v_{jl} \) in a tied set, assign a unique \( r_{jl}' \) by adding incremental multiples of \( \Delta r \) based on the secondary order:
    \[
    r_{jl}' = r_{jl} + (k - 1) \times \Delta r,
    \]
    where \( k \) is the position in the secondary ordering (starting from 1).
    
\end{enumerate}

\subparagraph{d. Final Numerical Representation}

The final numerical representation for each category \( v_{jl} \) is:
\[
r_{jl}' =
\begin{cases}
r_{jl} + (k - 1) \times \Delta r, & \text{if } v_{jl} \text{ is part of a tied set} \\
r_{jl}, & \text{otherwise},
\end{cases}
\]
ensuring that each category has a unique \( r_{jl}' \) value.

\subparagraph{e. Reverse Mapping: }

To facilitate efficient reverse mapping from numerical values \( r_{jl}' \) back to their original categorical values \( v_{jl} \), we implement a structured lookup mechanism. The process involves the following steps:

\begin{enumerate}
    \item \textbf{Construction of the Lookup Table:}
    
    During the encoding phase, alongside assigning each category its unique numerical representation \( r_{jl}' \), we construct a lookup table \( T_j \) for each categorical feature \( j \). The table \( T_j \) maps each \( r_{jl}' \) to its corresponding category \( v_{jl} \):
    \[
    T_j = \{ (r_{jl}', v_{jl}) \mid v_{jl} \in \mathcal{V}_j \}
    \]
    
    This table can be efficiently implemented using data structures such as hash tables or dictionaries, enabling constant-time \( O(1) \) access during reverse mapping.

    \item \textbf{Reverse Mapping Function:}
    
    To retrieve the original category from a given \( r_{jl}' \), the reverse mapping function performs the following:
    
    \begin{itemize}
        \item \textbf{Lookup Operation:} Given an \( r_{jl}' \), query the lookup table \( T_j \) to obtain the corresponding category \( v_{jl} \).
        
        \item \textbf{Handling Precision:} Ensure that the \( r_{jl}' \) values used during the attack or optimization process are matched exactly to those stored in \( T_j \). Implement rounding mechanisms if necessary to align floating-point representations.
    \end{itemize}
    
    Formally, the reverse mapping function \( RevConv(r_{jl}') \) is defined as:
    \[
    RevConv(r_{jl}') = v_{jl} \quad \text{such that} \quad (r_{jl}', v_{jl}) \in T_j
    \]
    
    \item \textbf{Optimization Considerations:}
    
    During optimization processes where \( r_{jl}' \) values might be adjusted continuously, it is crucial to maintain valid categorical representations. This is achieved by:
    
    \begin{itemize}
        \item \textbf{Rounding Adjusted Values:} Any continuous changes to \( r_{jl}' \) are rounded to the nearest valid value present in the lookup table \( T_j \):
        \[
        r_{jl}'^{\text{rounded}} = \text{round}(r_{jl}', \text{precision}=p')
        \]
        where \( p' \) corresponds to the decimal precision used in \( \Delta r \).
        
        \item \textbf{Validation:} Ensure that the rounded \( r_{jl}'^{\text{rounded}} \) exists within \( T_j \). If not, adjust \( r_{jl}' \) to the closest valid value to maintain consistency.
    \end{itemize}

\end{enumerate}

We also Define the function \(Revert(.)\), where it reverts back the whole dataset from converted numerical values to categorical values again.

\subparagraph{f. Example Illustration: }

Consider a categorical feature \( j \) with the following category counts:

\begin{table}[h]
    \centering
    \begin{tabular}{|c|c|}
        \hline
        \textbf{Category \( v_{jl} \)} & \textbf{Count \( c_{jl} \)} \\
        \hline
        A & 50 \\
        B & 50 \\
        C & 30 \\
        D & 20 \\
        \hline
    \end{tabular}
    \caption{Category Counts for Feature \( j \)}
    \label{tab:category_counts}
\end{table}

\textbf{Primary Mapping:}

\begin{align*}
r_{jA} &= \frac{50 - 50}{50 - 1} = 0.0000 \\
r_{jB} &= \frac{50 - 50}{50 - 1} = 0.0000 \\
r_{jC} &= \frac{50 - 30}{50 - 1} \approx 0.4082 \\
r_{jD} &= \frac{50 - 20}{50 - 1} \approx 0.6122 \\
\end{align*}


\textbf{Determine \( \Delta r \):}
\begin{itemize}
    \item \(\Delta r_{\text{min}} = 0.4082\)
    \item Identify the largest single decimal component in \(\Delta r_{\text{min}}\):
    \begin{itemize}
        \item \(\Delta r_{\text{min}} = 0.4082\) has the first non-zero decimal at the first decimal place (\( p = 1 \)).
    \end{itemize}
    \item Set \( \Delta r = 10^{-(p + 1)} = 10^{-2} = 0.01 \).
\end{itemize}

\textbf{Applying Hierarchical Mapping:}
\begin{align*}
r_{jA}' &= 0.0000 + (1 - 1) \times 0.01 = 0.0000 \\
r_{jB}' &= 0.0000 + (2 - 1) \times 0.01 = 0.0100 \\
r_{jC}' &= 0.4082 \\
r_{jD}' &= 0.6122 \\
\end{align*}

\textbf{Result:}

\begin{table}[h]
    \centering
    \begin{tabular}{|c|c|c|}
        \hline
        \textbf{Category \( v_{jl} \)} & \textbf{Original \( r_{jl} \)} & \textbf{Updated \( r_{jl}' \)} \\
        \hline
        A & 0.0000 & 0.0000 \\
        B & 0.0000 & 0.0100 \\
        C & 0.4082 & 0.4082 \\
        D & 0.6122 & 0.6122 \\
        \hline
    \end{tabular}
    \caption{Updated Numerical Representation After Hierarchical Mapping}
    \label{tab:updated_mapping}
\end{table}

 \textbf{Lookup Table:}
    
    Referring to Table~\ref{tab:updated_mapping}, the lookup table \( T_j \) for feature \( j \) is constructed as:
    
    \begin{table}[h]
        \centering
        \begin{tabular}{|c|c|}
            \hline
            \textbf{Numerical Value \( r_{jl}' \)} & \textbf{Category \( v_{jl} \)} \\
            \hline
            0.0000 & A \\
            0.0100 & B \\
            0.4082 & C \\
            0.6122 & D \\
            \hline
        \end{tabular}
        \caption{Lookup Table \( T_j \) for Feature \( j \)}
        \label{tab:lookup_table}
    \end{table}
    
    When an \( r_{jl}' \) value of 0.0100 is encountered during reverse mapping, the corresponding category retrieved from \( T_j \) is B.










\section{Attack Methodology}
\label{sec:attack_methodology}

The attack comprises the following steps:

\paragraph{1. Initial Model Training}

The categorical features in the dataset are transformed to numerical using the method explained in \autoref{sec:convert_cf}. Then, the attacker trains the model \( F \) on the converted dataset \( D = Conv(D_{original}) \) to obtain a baseline model that performs adequately on the classification task. 

\paragraph{2. Selection of Non-Target Samples}

The attacker constructs a subset \( D_{\text{non-target}} \) by excluding all samples with the target label \( t \):
\[
D_{\text{non-target}} = \{ (x_i, y_i) \in D \mid y_i \neq t \}.
\]

\paragraph{3. Confidence-Based Sample Ranking}

The attacker evaluates the trained model \( F \) on \( D_{\text{non-target}} \) to obtain the softmax confidence scores for the target class \( t \). For each input \( x_i \), the confidence score is:
\[
s_i = f_t(x_i),
\]
where \( f_t(x_i) \) is the softmax output corresponding to class \( t \).

The attacker pairs each input with its confidence score to form the set:
\[
D_{\text{conf}} = \{ (x_i, s_i) \mid (x_i, y_i) \in D_{\text{non-target}} \}.
\]

The attacker then sorts \( D_{\text{conf}} \) in descending order based on \( s_i \) and selects the top \( \mu \cdot |D_{\text{conf}}| \) samples to create the subset \( D_{\text{picked}} \), where \( \mu \in (0, 1] \) is a predefined fraction (e.g., \( \mu = 0.2 \)).

\paragraph{5. Definition of the Backdoor Trigger}

The attacker defines a universal trigger pattern \( \delta \in \mathbb{R}^d \) to be added to the inputs. The backdoored input \( \hat{x}_i \) is computed as:
\[
\hat{x}_i = \text{clip}( x_i + \delta ),
\]
where the clipping function ensures that each feature of \( \hat{x}_i \) remains within its valid range:
\[
\hat{x}_i^{(j)} = \begin{cases}
\max X^{(j)}, & \text{if } x_i^{(j)} + \delta^{(j)} > \max X^{(j)}, \\
\min X^{(j)}, & \text{if } x_i^{(j)} + \delta^{(j)} < \min X^{(j)}, \\
x_i^{(j)} + \delta^{(j)}, & \text{otherwise},
\end{cases}
\]
with \( \min X^{(j)} \) and \( \max X^{(j)} \) being the minimum and maximum values of feature \( j \) in \( D \).

\paragraph{6. Optimization of the Trigger Pattern}

The attacker optimizes \( \delta \) by minimizing the following loss function over \( D_{\text{picked}} \):
\begin{align*}
\mathcal{L}(\delta) = \frac{1}{|D_{\text{picked}}|} \sum_{(x_i, y_i) \in D_{\text{picked}}} & \left[ -\log f_t( \hat{x}_i ) + \beta \| \hat{x}_i - \text{Mode}(X) \|_1 \right. \\
& \left. + \lambda \| \hat{x}_i - \text{Mode}(X) \|_2^2 \right],
\end{align*}


where:
\begin{itemize}
    \item \( f_t( \hat{x}_i ) \) is the softmax output for class \( t \) given input \( \hat{x}_i \).
    \item \( \text{Mode}(X) \in \mathbb{R}^d \) is the mode vector of the dataset \( D \), with each element \( \text{Mode}(X)^{(j)} \) being the mode of feature \( j \).
    \item \( \beta \) and \( \lambda \) are hyperparameters controlling the \( L_1 \) and \( L_2 \) regularization terms, respectively.
\end{itemize}

The loss function balances three objectives:
\begin{enumerate}
    \item Maximizing the model's confidence in predicting the target class \( t \) for the backdoored inputs.
    \item Ensuring the trigger pattern \( \delta \) keeps the modified inputs close to common data patterns (via the mode) to enhance stealthiness.
    \item Regularizing \( \delta \) to prevent large perturbations that could be easily detected.
\end{enumerate}

The optimal trigger pattern \( \delta^* \) is obtained by solving:
\[
\delta^* = \arg\min_{\delta} \mathcal{L}(\delta).
\]

This optimization is performed using gradient descent, updating \( \delta \) iteratively based on the gradient \( \nabla_\delta \mathcal{L} \). Note that in each round after modifications, we use \(r_{jl}'^{\text{rounded}}\) (as explained in \autoref{sec:convert_cf}) to round the modified \(r_{jl}\) values.

\paragraph{7. Construction of the Poisoned Dataset}

With \( \delta^* \) optimized, the attacker selects randomly a fraction \( \epsilon \in (0, 1] \) of the dataset \( D \) to poison.

% The selection can be random or based on specific criteria (e.g., samples where the attack is more likely to succeed).

Each selected sample \( (x_i, y_i) \) is modified:
\[
\hat{x}_i = \text{clip}( x_i + \delta^* ), \quad \hat{y}_i = t.
\]

The poisoned dataset \( D_{\text{poisoned}} \) consists of these modified samples:
\[
D_{\text{poisoned}} = \{ (\hat{x}_i, \hat{y}_i) \mid (x_i, y_i) \in D_{\text{selected}} \},
\]
where \( D_{\text{selected}} \subset D \) and \( |D_{\text{selected}}| = \epsilon \cdot N \).

The final training dataset is:
\[
D' = Revert(\left( D \setminus D_{\text{selected}} \right) \cup D_{\text{poisoned}}).
\]

\paragraph{8. Training the Backdoored Model}

The attacker retrains the model \( F' \) on the poisoned dataset \( D' \). 

To handle categorical features in \( D' \), the attacker adheres to standard preprocessing protocols that an innocent user would typically employ. This includes utilizing encoding techniques such as embedding methods (with random or Xavier initialization) or one-hot encoding to transform categorical variables into a numerical format suitable for neural network training. In this study, we employ the \texttt{OrdinalEncoder} from scikit-learn to convert all categorical features into ordinal integers.

The expectation is that \( F' \) maintains performance on clean data while exhibiting the backdoor behavior when the trigger is present.

\paragraph{9. Deployment and Attack Activation}

During deployment, any input \( x \) modified with the trigger pattern \( \delta^* \) will be misclassified as the target label \( t \):
\[
F'( \hat{x} ) = F'( x + \delta^* ) = t.
\]

\subsection{Discussion}

Our methodology demonstrates how categorical features can be exploited in backdoor attacks on tabular data. By mapping categorical features to numerical values based on category frequencies, we create a seamless integration of the trigger pattern across all features. The optimization of \( \delta \) ensures that the trigger is both effective and stealthy, minimizing detection by keeping the perturbed inputs close to common data patterns (as represented by the mode vector).

\subsection{Hyperparameter Considerations}

The hyperparameters \( \mu \), \( \beta \), \( \lambda \), and \( \epsilon \) play crucial roles in the attack:

\begin{itemize}
    \item \( \mu \) (\( 0 < \mu \leq 1 \)) controls the proportion of high-confidence samples used for optimizing \( \delta \). A higher \( \mu \) may lead to a more generalized trigger but could also increase optimization difficulty.
    \item \( \beta \) and \( \lambda \) (\( \beta, \lambda > 0 \)) regulate the emphasis on stealthiness versus attack efficacy. They should be set to balance minimizing perturbations and maximizing the model's confidence in the target class.
    \item \( \epsilon \) (\( 0 < \epsilon \leq 1 \)) determines the fraction of the dataset to poison. A smaller \( \epsilon \) enhances stealth but may reduce the backdoor's effectiveness.
\end{itemize}

These parameters can be tuned based on experimental results to achieve the desired trade-offs between attack success rate, stealthiness, and impact on model performance.
