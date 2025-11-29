body: JSON.stringify(formData)
            });

if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
}

const data = await response.json();

if (data.status === 'success') {
    // Show all steps as completed
    completeAllSteps();

    // Display results after a short delay
    setTimeout(() => {
        displayResults(data);
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });
    }, 1000);
} else {
    throw new Error(data.message || 'Analysis failed');
}

        } catch (error) {
    console.error('Error:', error);
    alert(`Error: ${error.message}\n\nPlease try again or check the console for details.`);
} finally {
    // Re-enable submit button
    analyzeBtn.disabled = false;
    analyzeBtn.innerHTML = '<span class="btn-icon">⚡</span> Start Complete Analysis';
}
    });

// Simulate progress animation
function animateProgress() {
    const steps = document.querySelectorAll('.progress-step');
    let currentStep = 0;

    const interval = setInterval(() => {
        if (currentStep < steps.length) {
            steps[currentStep].classList.add('active');
            if (currentStep > 0) {
                steps[currentStep - 1].classList.remove('active');
                steps[currentStep - 1].classList.add('completed');
                steps[currentStep - 1].querySelector('.step-status').textContent = '✅';
            }
            currentStep++;
        }
    }, 2000);

    return interval;
}

function resetProgressSteps() {
    const steps = document.querySelectorAll('.progress-step');
    steps.forEach(step => {
        step.classList.remove('active', 'completed');
        step.querySelector('.step-status').textContent = '⏳';
    });
}

function completeAllSteps() {
    const steps = document.querySelectorAll('.progress-step');
    steps.forEach(step => {
        step.classList.remove('active');
        step.classList.add('completed');
        step.querySelector('.step-status').textContent = '✅';
    });
}

function getAIBadge(aiTechnique, isAIPowered) {
    if (!isAIPowered) {
        return '<div class="ai-badge fallback">⚠️ Fallback Mode</div>';
    }
    return `<div class="ai-badge">${aiTechnique}</div>`;
}

function displayResults(data) {
    resultsContainer.innerHTML = '';

    // Module 1: Scraper Results
    if (data.module_1_scraper) {
        const scraperCard = createResultCard(
            '📊 Marketplace Data',
            `${getAIBadge(data.module_1_scraper.ai_technique, data.module_1_scraper.is_ai_powered)}
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Products Scraped</div>
                        <div class="stat-value">${data.module_1_scraper.total_products_scraped}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Platforms</div>
                        <div class="stat-value">${data.module_1_scraper.platforms.join(', ')}</div>
                    </div>
                </div>`
        );
        resultsContainer.appendChild(scraperCard);
    }

    // Module 2: Trends
    if (data.module_2_trends && data.module_2_trends.category_trends) {
        const trends = data.module_2_trends.category_trends;
        const forecast = data.module_2_trends.demand_forecast;

        const trendsCard = createResultCard(
            '📈 Market Trends & Forecast',
            `${getAIBadge(data.module_2_trends.ai_technique, data.module_2_trends.is_ai_powered)}
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Average Price</div>
                        <div class="stat-value">$${trends.average_price?.toFixed(2) || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Average Rating</div>
                        <div class="stat-value">${trends.average_rating?.toFixed(1) || 'N/A'}/5.0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Demand Level</div>
                        <div class="stat-value">${forecast?.demand_level || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Demand Score</div>
                        <div class="stat-value">${forecast?.current_demand_score || 'N/A'}/100</div>
                    </div>
                </div>
                ${forecast?.recommendations ? `
                    <h4 style="margin-top: 1.5rem; margin-bottom: 0.75rem; color: var(--text-secondary);">Recommendations:</h4>
                    ${forecast.recommendations.map(rec => `<div class="list-item">${rec}</div>`).join('')}
                ` : ''}`
        );
        resultsContainer.appendChild(trendsCard);
    }

    // Module 3: Competitors
    if (data.module_3_competitors) {
        const comp = data.module_3_competitors;
        const analysis = comp.competitive_analysis;

        const competitorsCard = createResultCard(
            '🎯 Competitor Analysis',
            `${getAIBadge(data.module_3_competitors.ai_technique, data.module_3_competitors.is_ai_powered)}
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Competitors Found</div>
                        <div class="stat-value">${comp.total_competitors}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Market Average Price</div>
                        <div class="stat-value">$${analysis?.market_statistics?.average_price?.toFixed(2) || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Your Position</div>
                        <div class="stat-value">${analysis?.price_positioning || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Market Saturation</div>
                        <div class="stat-value">${analysis?.market_saturation?.level || 'N/A'}</div>
                    </div>
                </div>
                ${analysis?.recommendations ? `
                    <h4 style="margin-top: 1.5rem; margin-bottom: 0.75rem; color: var(--text-secondary);">Strategic Recommendations:</h4>
                    ${analysis.recommendations.slice(0, 5).map(rec => `<div class="list-item">${rec}</div>`).join('')}
                ` : ''}`
        );
        resultsContainer.appendChild(competitorsCard);
    }

    // Module 4: Suppliers
    if (data.module_4_suppliers && data.module_4_suppliers.recommended_suppliers) {
        const suppliers = data.module_4_suppliers.recommended_suppliers;

        const suppliersCard = createResultCard(
            '🏭 Recommended Suppliers',
            `${getAIBadge(data.module_4_suppliers.ai_technique, data.module_4_suppliers.is_ai_powered)}
                ${suppliers.map((supplier, index) => `
                    <div class="list-item" style="margin-bottom: 1rem;">
                        <strong>#${index + 1}: ${supplier.name}</strong><br>
                        <small style="color: var(--text-muted);">
                            ${supplier.type} | ${supplier.region}<br>
                            Score: ${supplier.recommendation_score}/100 | 
                            Est. Cost: $${supplier.estimated_unit_cost?.toFixed(2)}/unit<br>
                            MOQ: ${supplier.moq_range[0]}-${supplier.moq_range[1]} units
                        </small>
                    </div>
                `).join('')}`
        );
        resultsContainer.appendChild(suppliersCard);
    }

    // Module 5: Pricing
    if (data.module_5_pricing) {
        const pricing = data.module_5_pricing;

        const pricingCard = createResultCard(
            '💰 Pricing & Profitability',
            `${getAIBadge(data.module_5_pricing.ai_technique, data.module_5_pricing.is_ai_powered)}
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Product Cost</div>
                        <div class="stat-value">$${pricing.product_cost?.toFixed(2)}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Amazon Price</div>
                        <div class="stat-value">$${pricing.amazon_pricing?.recommended_price?.toFixed(2) || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Amazon Margin</div>
                        <div class="stat-value">${pricing.amazon_pricing?.profit_margin?.toFixed(1) || 'N/A'}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Best Platform</div>
                        <div class="stat-value">${pricing.platform_comparison[0]?.platform || 'N/A'}</div>
                    </div>
                </div>`
        );
        resultsContainer.appendChild(scraperCard);
    }

    // Module 2: Trends
    if (data.module_2_trends && data.module_2_trends.category_trends) {
        const trends = data.module_2_trends.category_trends;
        const forecast = data.module_2_trends.demand_forecast;
        const accuracy = trends.accuracy_metrics || forecast?.accuracy_metrics || {};

        const trendsCard = createResultCard(
            '📈 Market Trends & Forecast',
            `${getAccuracyBadge(accuracy)}
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Average Price</div>
                        <div class="stat-value">$${trends.average_price?.toFixed(2) || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Average Rating</div>
                        <div class="stat-value">${trends.average_rating?.toFixed(1) || 'N/A'}/5.0</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Demand Level</div>
                        <div class="stat-value">${forecast?.demand_level || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Demand Score</div>
                        <div class="stat-value">${forecast?.current_demand_score || 'N/A'}/100</div>
                    </div>
                </div>
                ${forecast?.recommendations ? `
                    <h4 style="margin-top: 1.5rem; margin-bottom: 0.75rem; color: var(--text-secondary);">Recommendations:</h4>
                    ${forecast.recommendations.map(rec => `<div class="list-item">${rec}</div>`).join('')}
                ` : ''}`
        );
        resultsContainer.appendChild(trendsCard);
    }

    // Module 3: Competitors
    if (data.module_3_competitors) {
        const comp = data.module_3_competitors;
        const analysis = comp.competitive_analysis;
        const accuracy = analysis?.accuracy_metrics || {};

        const competitorsCard = createResultCard(
            '🎯 Competitor Analysis',
            `${getAccuracyBadge(accuracy)}
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Competitors Found</div>
                        <div class="stat-value">${comp.total_competitors}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Market Average Price</div>
                        <div class="stat-value">$${analysis?.market_statistics?.average_price?.toFixed(2) || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Your Position</div>
                        <div class="stat-value">${analysis?.price_positioning || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Market Saturation</div>
                        <div class="stat-value">${analysis?.market_saturation?.level || 'N/A'}</div>
                    </div>
                </div>
                ${analysis?.recommendations ? `
                    <h4 style="margin-top: 1.5rem; margin-bottom: 0.75rem; color: var(--text-secondary);">Strategic Recommendations:</h4>
                    ${analysis.recommendations.slice(0, 5).map(rec => `<div class="list-item">${rec}</div>`).join('')}
                ` : ''}`
        );
        resultsContainer.appendChild(competitorsCard);
    }

    // Module 4: Suppliers
    if (data.module_4_suppliers && data.module_4_suppliers.recommended_suppliers) {
        const suppliers = data.module_4_suppliers.recommended_suppliers;
        const accuracy = data.module_4_suppliers.accuracy_metrics || {};

        const suppliersCard = createResultCard(
            '🏭 Recommended Suppliers',
            `${getAccuracyBadge(accuracy)}
                ${suppliers.map((supplier, index) => `
                    <div class="list-item" style="margin-bottom: 1rem;">
                        <strong>#${index + 1}: ${supplier.name}</strong><br>
                        <small style="color: var(--text-muted);">
                            ${supplier.type} | ${supplier.region}<br>
                            Score: ${supplier.recommendation_score}/100 | 
                            Est. Cost: $${supplier.estimated_unit_cost?.toFixed(2)}/unit<br>
                            MOQ: ${supplier.moq_range[0]}-${supplier.moq_range[1]} units
                        </small>
                    </div>
                `).join('')}`
        );
        resultsContainer.appendChild(suppliersCard);
    }

    // Module 5: Pricing
    if (data.module_5_pricing) {
        const pricing = data.module_5_pricing;
        const accuracy = pricing.amazon_pricing?.accuracy_metrics || {};

        const pricingCard = createResultCard(
            '💰 Pricing & Profitability',
            `${getAccuracyBadge(accuracy)}
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Product Cost</div>
                        <div class="stat-value">$${pricing.product_cost?.toFixed(2)}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Amazon Price</div>
                        <div class="stat-value">$${pricing.amazon_pricing?.recommended_price?.toFixed(2) || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Amazon Margin</div>
                        <div class="stat-value">${pricing.amazon_pricing?.profit_margin?.toFixed(1) || 'N/A'}%</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Best Platform</div>
                        <div class="stat-value">${pricing.platform_comparison[0]?.platform || 'N/A'}</div>
                    </div>
                </div>`
        );
        resultsContainer.appendChild(pricingCard);
    }

    // Module 6: Platforms
    if (data.module_6_platforms && data.module_6_platforms.recommended_platforms) {
        const platforms = data.module_6_platforms.recommended_platforms;

        const platformsCard = createResultCard(
            '🛒 Platform Recommendations',
            platforms.slice(0, 3).map((platform, index) => `
                    <div class="list-item" style="margin-bottom: 1rem;">
                        ${index === 0 ? getAccuracyBadge(platform.accuracy_metrics || {}) : ''}
                        <strong>#${index + 1}: ${platform.platform_name}</strong><br>
                        <small style="color: var(--text-muted);">
                            Suitability: ${platform.suitability_rating} (Score: ${platform.recommendation_score}/100)<br>
                            Setup: ${platform.setup_difficulty?.level || 'N/A'} | 
                            Time: ${platform.setup_difficulty?.time_to_setup || 'N/A'}
                        </small>
                    </div>
                `).join('')
        );
        resultsContainer.appendChild(platformsCard);
    }

    // Module 7: Audience
    if (data.module_7_audience && data.module_7_audience.audience_profile) {
        const audience = data.module_7_audience.audience_profile;
        const demographics = audience.primary_demographics;
        const personas = audience.buyer_personas;
        const accuracy = audience.accuracy_metrics || {};

        const audienceCard = createResultCard(
            '👥 Target Audience Profile',
            `${getAccuracyBadge(accuracy)}
                <div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Age Range</div>
                        <div class="stat-value">${demographics?.age_range || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Income Level</div>
                        <div class="stat-value">${demographics?.income_level || 'N/A'}</div>
                    </div>
                </div>
                ${personas && personas.length > 0 ? `
                    <h4 style="margin-top: 1.5rem; margin-bottom: 0.75rem; color: var(--text-secondary);">Primary Persona: ${personas[0].name}</h4>
                    <div class="list-item">
                        <strong>Occupation:</strong> ${personas[0].occupation}<br>
                        <strong>Goals:</strong> ${personas[0].goals?.slice(0, 2).join(', ')}
                    </div>
                ` : ''}`
        );
        resultsContainer.appendChild(audienceCard);
    }

    // Module 8: Marketing
    if (data.module_8_marketing && data.module_8_marketing.marketing_strategy) {
        const marketing = data.module_8_marketing.marketing_strategy;
        const overview = marketing.overview;
        const channels = marketing.channel_strategy?.channels || [];
        const accuracy = marketing.accuracy_metrics || {};

        const marketingCard = createResultCard(
            '📱 Marketing Strategy',
            `${getAccuracyBadge(accuracy)}
                <div class="stat-grid">`
        );
        resultsContainer.appendChild(pricingCard);
    }

    // Module 6: Platforms
    if (data.module_6_platforms && data.module_6_platforms.recommended_platforms) {
        const platforms = data.module_6_platforms.recommended_platforms;

        const platformsCard = createResultCard(
            '🛒 Platform Recommendations',
            platforms.slice(0, 3).map((platform, index) => `
                    <div class="list-item" style="margin-bottom: 1rem;">
                        <strong>#${index + 1}: ${platform.platform_name}</strong><br>
                        <small style="color: var(--text-muted);">
                            Suitability: ${platform.suitability_rating} (Score: ${platform.recommendation_score}/100)<br>
                            Setup: ${platform.setup_difficulty?.level || 'N/A'} | 
                            Time: ${platform.setup_difficulty?.time_to_setup || 'N/A'}
                        </small>
                    </div>
                `).join('')
        );
        resultsContainer.appendChild(platformsCard);
    }

    // Module 7: Audience
    if (data.module_7_audience && data.module_7_audience.audience_profile) {
        const audience = data.module_7_audience.audience_profile;
        const demographics = audience.primary_demographics;
        const personas = audience.buyer_personas;

        const audienceCard = createResultCard(
            '👥 Target Audience Profile',
            `<div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Age Range</div>
                        <div class="stat-value">${demographics?.age_range || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Income Level</div>
                        <div class="stat-value">${demographics?.income_level || 'N/A'}</div>
                    </div>
                </div>
                ${personas && personas.length > 0 ? `
                    <h4 style="margin-top: 1.5rem; margin-bottom: 0.75rem; color: var(--text-secondary);">Primary Persona: ${personas[0].name}</h4>
                    <div class="list-item">
                        <strong>Occupation:</strong> ${personas[0].occupation}<br>
                        <strong>Goals:</strong> ${personas[0].goals?.slice(0, 2).join(', ')}
                    </div>
                ` : ''}`
        );
        resultsContainer.appendChild(audienceCard);
    }

    // Module 8: Marketing
    if (data.module_8_marketing && data.module_8_marketing.marketing_strategy) {
        const marketing = data.module_8_marketing.marketing_strategy;
        const overview = marketing.overview;
        const channels = marketing.channel_strategy?.channels || [];

        const marketingCard = createResultCard(
            '📱 Marketing Strategy',
            `<div class="stat-grid">
                    <div class="stat-item">
                        <div class="stat-label">Budget</div>
                        <div class="stat-value">$${overview?.total_budget || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Budget Tier</div>
                        <div class="stat-value">${overview?.budget_tier || 'N/A'}</div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Strategy Period</div>
                        <div class="stat-value">${overview?.strategy_period || 'N/A'}</div>
                    </div>
                </div>
                <h4 style="margin-top: 1.5rem; margin-bottom: 0.75rem; color: var(--text-secondary);">Key Channels:</h4>
                ${channels.slice(0, 3).map(channel => `
                    <div class="list-item">
                        <strong>${channel.channel}</strong> (${channel.budget_allocation})<br>
                        <small style="color: var(--text-muted);">Priority: ${channel.priority}</small>
                    </div>
                `).join('')}`
        );
        resultsContainer.appendChild(marketingCard);
    }

    // Success message and download option
    const successCard = document.createElement('div');
    successCard.className = 'result-card fade-in';
    successCard.innerHTML = `
            <h3 style="color: var(--success);"><span class="icon">✅</span> Analysis Complete!</h3>
            <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                Your comprehensive business analysis has been generated. All modules have been executed successfully.
            </p>
            <p style="color: var(--text-muted); font-size: 0.875rem;">
                Results saved to: ${data.results_file || 'warehouse directory'}
            </p>
        `;
    resultsContainer.appendChild(successCard);
}

function createResultCard(title, content) {
    const card = document.createElement('div');
    card.className = 'result-card fade-in';
    card.innerHTML = `
            <h3><span class="icon">${title.split(' ')[0]}</span> ${title.substring(title.indexOf(' ') + 1)}</h3>
            ${content}
        `;
    return card;
}

function getAccuracyBadge(accuracy) {
    if (!accuracy || !accuracy.confidence_score) {
        return '';
    }

    const score = accuracy.confidence_score;
    const level = accuracy.confidence_level || 'Unknown';

    // Determine badge color based on confidence level
    let badgeColor = '#64748b'; // default gray
    if (level === 'High') {
        badgeColor = '#10b981'; // green
    } else if (level === 'Medium') {
        badgeColor = '#f59e0b'; // orange
    } else if (level === 'Low') {
        badgeColor = '#ef4444'; // red
    }

    return `
            <div style="display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; padding: 0.75rem; background: rgba(255,255,255,0.05); border-radius: 8px;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-weight: 600; color: var(--text-secondary); font-size: 0.875rem;">Accuracy:</span>
                    <span style="background: ${badgeColor}; color: white; padding: 0.25rem 0.75rem; border-radius: 12px; font-size: 0.875rem; font-weight: 600;">
                        ${level} (${score}%)
                    </span>
                </div>
                ${accuracy.recommendation_reliability || accuracy.analysis_reliability || accuracy.reliability ? `
                    <span style="color: var(--text-muted); font-size: 0.8rem; flex: 1;">
                        ${accuracy.recommendation_reliability || accuracy.analysis_reliability || accuracy.reliability}
                    </span>
                ` : ''}
            </div>
        `;
}
});
